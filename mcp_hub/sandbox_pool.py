"""
Warm Sandbox Pool for Modal - Async Queue-Based Implementation
This module provides a pre-warmed pool of Modal sandboxes to reduce cold-start latency.
"""
import asyncio
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

import modal

from mcp_hub.logging_config import logger
from mcp_hub.exceptions import CodeExecutionError


class SandboxHealth(Enum):
    """Sandbox health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class PooledSandbox:
    """Container for a pooled sandbox with metadata."""
    sandbox: modal.Sandbox
    created_at: float
    last_used: float
    health: SandboxHealth = SandboxHealth.UNKNOWN
    use_count: int = 0


class WarmSandboxPool:
    """Async queue-based warm sandbox pool with health checking."""
    
    def __init__(
        self,
        app: modal.App,
        image: modal.Image,
        pool_size: int = 3,
        max_age_seconds: int = 300,  # 5 minutes
        max_uses_per_sandbox: int = 10,
        health_check_interval: int = 60,  # 1 minute
    ):
        self.app = app
        self.image = image
        self.pool_size = pool_size
        self.max_age_seconds = max_age_seconds
        self.max_uses_per_sandbox = max_uses_per_sandbox
        self.health_check_interval = health_check_interval
        
        # Queue to hold available sandboxes
        self._sandbox_queue: asyncio.Queue[PooledSandbox] = asyncio.Queue(maxsize=pool_size)
        
        # Background tasks
        self._warmup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Pool statistics
        self._stats = {
            "created": 0,
            "reused": 0,
            "recycled": 0,
            "health_checks": 0,
            "failures": 0
        }
        
        self._running = False
        
    async def start(self):
        """Start the pool and background tasks."""
        if self._running:
            return
            
        self._running = True
        logger.info(f"Starting warm sandbox pool with {self.pool_size} sandboxes")
        
        # Start background tasks
        self._warmup_task = asyncio.create_task(self._warmup_pool())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Wait for initial warmup
        await asyncio.sleep(1)  # Give warmup a moment to start
        
    async def stop(self):
        """Stop the pool and cleanup resources."""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping warm sandbox pool")
        
        # Cancel background tasks
        for task in [self._warmup_task, self._health_check_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cleanup remaining sandboxes
        while not self._sandbox_queue.empty():
            try:
                pooled_sb = self._sandbox_queue.get_nowait()
                await self._terminate_sandbox(pooled_sb.sandbox)
            except asyncio.QueueEmpty:
                break
                
    @asynccontextmanager
    async def get_sandbox(self, timeout: float = 5.0):
        pooled_sb = None
        created_new = False
        try:
            # Try to get a warm sandbox from the pool, retry if not alive
            max_retries = 2
            for _ in range(max_retries):
                try:
                    pooled_sb = await asyncio.wait_for(self._sandbox_queue.get(), timeout=timeout)
                    # Check if the sandbox is alive
                    alive = await self._is_sandbox_alive(pooled_sb.sandbox)
                    if not alive:
                        logger.info("Got dead sandbox from pool, terminating and trying next.")
                        await self._terminate_sandbox(pooled_sb.sandbox)
                        continue  # Try again
                    pooled_sb.last_used = time.time()
                    pooled_sb.use_count += 1
                    self._stats["reused"] += 1
                    break
                except asyncio.TimeoutError:
                    # Pool empty, create a new one
                    logger.info("Pool empty, creating new sandbox")
                    sandbox = await self._create_sandbox()
                    pooled_sb = PooledSandbox(
                        sandbox=sandbox,
                        created_at=time.time(),
                        last_used=time.time(),
                        use_count=1
                    )
                    created_new = True
                    self._stats["created"] += 1
                    break
            else:
                raise CodeExecutionError("Could not obtain a live sandbox from the pool.")
            logger.info(f"Yielding sandbox of type from sandbox_pool: {type(pooled_sb.sandbox)}")    
            yield pooled_sb.sandbox
        except Exception as e:
            logger.error(f"Error getting sandbox: {e}")
            self._stats["failures"] += 1
            raise CodeExecutionError(f"Failed to get sandbox: {e}")
        finally:
            if pooled_sb:
                should_recycle = (
                    not created_new and
                    self._should_recycle_sandbox(pooled_sb) and
                    self._running
                )
                if should_recycle:
                    # Check alive before returning to pool
                    if await self._is_sandbox_alive(pooled_sb.sandbox):
                        try:
                            self._sandbox_queue.put_nowait(pooled_sb)
                            logger.debug("Returned sandbox to pool")
                        except asyncio.QueueFull:
                            # TERMINATE if pool is full
                            await self._terminate_sandbox(pooled_sb.sandbox)
                            logger.debug("Pool full, terminated sandbox")
                    else:
                        # TERMINATE dead sandbox
                        await self._terminate_sandbox(pooled_sb.sandbox)
                        logger.debug("Not returning dead sandbox to pool")
                else:
                    # TERMINATE
                    await self._terminate_sandbox(pooled_sb.sandbox)
                    if not created_new:
                        self._stats["recycled"] += 1
    
    async def _create_sandbox(self) -> modal.Sandbox:
        """Create a new Modal sandbox."""
        try:
            sandbox = modal.Sandbox.create(
                app=self.app,
                image=self.image,
                cpu=2.0,
                memory=1024,
                timeout=35
            )
            logger.debug("Created new sandbox")
            logger.debug(f"Created new sandbox, type: {type(sandbox)}")
            return sandbox
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise
    
    async def _terminate_sandbox(self, sandbox: modal.Sandbox):
        """Safely terminate a sandbox."""
        try:
            await asyncio.get_event_loop().run_in_executor(None, sandbox.terminate)
            logger.debug("Terminated sandbox")
        except Exception as e:
            logger.warning(f"Failed to terminate sandbox: {e}")
    
    def _should_recycle_sandbox(self, pooled_sb: PooledSandbox) -> bool:
        """Determine if a sandbox should be recycled back to the pool."""
        now = time.time()
        
        # Check age
        if now - pooled_sb.created_at > self.max_age_seconds:
            logger.debug("Sandbox too old, not recycling")
            return False
            
        # Check usage count
        if pooled_sb.use_count >= self.max_uses_per_sandbox:
            logger.debug("Sandbox used too many times, not recycling")
            return False
            
        # Check health (if we've checked it)
        if pooled_sb.health == SandboxHealth.UNHEALTHY:
            logger.debug("Sandbox unhealthy, not recycling")
            return False
            
        return True
    
    async def _warmup_pool(self):
        """Background task to maintain warm sandboxes in the pool."""
        while self._running:
            try:
                current_size = self._sandbox_queue.qsize()
                if current_size < self.pool_size:
                    # Create new sandboxes to fill the pool
                    tasks = []
                    for _ in range(self.pool_size - current_size):
                        task = asyncio.create_task(self._create_and_queue_sandbox())
                        tasks.append(task)
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                        
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in warmup loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _create_and_queue_sandbox(self):
        """Create a sandbox and add it to the queue."""
        try:
            sandbox = await self._create_sandbox()
            pooled_sb = PooledSandbox(
                sandbox=sandbox,
                created_at=time.time(),
                last_used=time.time()
            )
            
            try:
                self._sandbox_queue.put_nowait(pooled_sb)
                logger.debug("Added warm sandbox to pool")
            except asyncio.QueueFull:
                # Pool is full, terminate this sandbox
                await self._terminate_sandbox(sandbox)
                
        except Exception as e:
            logger.error(f"Failed to create and queue sandbox: {e}")
    
    async def _health_check_loop(self):
        """Background task to check sandbox health."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on sandboxes in the pool."""
        # This is a simplified health check - in practice you might want
        # to run a simple command to verify the sandbox is responsive
        temp_sandboxes = []
        
        # Drain the queue to check each sandbox
        while not self._sandbox_queue.empty():
            try:
                pooled_sb = self._sandbox_queue.get_nowait()
                is_healthy = await self._check_sandbox_health(pooled_sb.sandbox)
                pooled_sb.health = SandboxHealth.HEALTHY if is_healthy else SandboxHealth.UNHEALTHY
                if is_healthy:
                    temp_sandboxes.append(pooled_sb)
                else:
                    # TERMINATE unhealthy sandbox
                    await self._terminate_sandbox(pooled_sb.sandbox)
                    self._stats["recycled"] += 1
            except asyncio.QueueEmpty:
                break
        
        # Put healthy sandboxes back
        for pooled_sb in temp_sandboxes:
            try:
                self._sandbox_queue.put_nowait(pooled_sb)
            except asyncio.QueueFull:
                await self._terminate_sandbox(pooled_sb.sandbox)
        
        self._stats["health_checks"] += 1
        logger.debug(f"Health check completed. Pool size: {self._sandbox_queue.qsize()}")
    
    async def _check_sandbox_health(self, sandbox: modal.Sandbox) -> bool:
        """Check if a sandbox is healthy."""
        try:
            # Run a simple Python command to check if the sandbox is responsive
            proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sandbox.exec("python", "-c", "print('health_check')", timeout=5)
            )
            output = proc.stdout.read()
            return "health_check" in output
        except Exception as e:
            logger.debug(f"Sandbox health check failed: {e}")
            return False
    
    async def _cleanup_loop(self):
        """Background task to cleanup old sandboxes."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._cleanup_old_sandboxes()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_old_sandboxes(self):
        """Remove old sandboxes from the pool."""
        now = time.time()
        temp_sandboxes = []
        
        while not self._sandbox_queue.empty():
            try:
                pooled_sb = self._sandbox_queue.get_nowait()
                if now - pooled_sb.created_at < self.max_age_seconds:
                    temp_sandboxes.append(pooled_sb)
                else:
                    # TERMINATE expired sandbox
                    await self._terminate_sandbox(pooled_sb.sandbox)
                    self._stats["recycled"] += 1
                    logger.debug("Cleaned up old sandbox")
            except asyncio.QueueEmpty:
                break
        
        # Put non-expired sandboxes back
        for pooled_sb in temp_sandboxes:
            try:
                self._sandbox_queue.put_nowait(pooled_sb)
            except asyncio.QueueFull:
                await self._terminate_sandbox(pooled_sb.sandbox)

    async def _is_sandbox_alive(self, sandbox: modal.Sandbox) -> bool:
        """Check if a sandbox is alive by running a trivial command."""
        try:
            proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sandbox.exec("python", "-c", "print('ping')", timeout=3)
            )
            out = proc.stdout.read() if hasattr(proc.stdout, "read") else str(proc)
            return "ping" in out
        except Exception as e:
            logger.debug(f"Liveness check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self._stats,
            "pool_size": self._sandbox_queue.qsize(),
            "target_pool_size": self.pool_size,
            "running": self._running
        }
