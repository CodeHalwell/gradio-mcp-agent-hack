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
        pool_size: int = 6,
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
        
        # Health tracking for better error recovery
        self._consecutive_failures = 0
        self._last_successful_creation = time.time()
        self._pool_reset_threshold = 5  # Reset pool after 5 consecutive failures
        
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
            # Check if we need to reset the pool due to consecutive failures
            if self._consecutive_failures >= self._pool_reset_threshold:
                logger.warning(f"Pool has {self._consecutive_failures} consecutive failures, attempting reset")
                await self._emergency_pool_reset()
            
            # Try to get a warm sandbox from the pool, retry if not alive
            max_retries = 3  # Increased retries for better reliability
            for attempt in range(max_retries):
                try:
                    # Try to get from pool first
                    pooled_sb = await asyncio.wait_for(self._sandbox_queue.get(), timeout=timeout)
                    # Check if the sandbox is alive
                    alive = await self._is_sandbox_alive(pooled_sb.sandbox)
                    if not alive:
                        logger.info(f"Got dead sandbox from pool on attempt {attempt + 1}, terminating and trying next.")
                        await self._terminate_sandbox(pooled_sb.sandbox)
                        pooled_sb = None
                        continue  # Try again
                    
                    # Sandbox is alive, use it
                    pooled_sb.last_used = time.time()
                    pooled_sb.use_count += 1
                    self._stats["reused"] += 1
                    self._consecutive_failures = 0  # Reset failure counter on success
                    break
                    
                except asyncio.TimeoutError:
                    # Pool empty or taking too long, create a new one
                    logger.info(f"Pool timeout on attempt {attempt + 1}, creating new sandbox")
                    try:
                        sandbox = await self._create_sandbox()
                        pooled_sb = PooledSandbox(
                            sandbox=sandbox,
                            created_at=time.time(),
                            last_used=time.time(),
                            use_count=1
                        )
                        created_new = True
                        self._stats["created"] += 1
                        self._consecutive_failures = 0  # Reset failure counter on success
                        self._last_successful_creation = time.time()
                        break
                    except Exception as create_error:
                        logger.error(f"Failed to create sandbox on attempt {attempt + 1}: {create_error}")
                        self._consecutive_failures += 1
                        if attempt == max_retries - 1:  # Last attempt
                            raise CodeExecutionError(f"Failed to create sandbox after {max_retries} attempts: {create_error}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                self._consecutive_failures += 1
                raise CodeExecutionError("Could not obtain a live sandbox from the pool after all retry attempts.")
            
            logger.info(f"Yielding sandbox of type from sandbox_pool: {type(pooled_sb.sandbox)}")    
            yield pooled_sb.sandbox
            
        except Exception as e:
            logger.error(f"Error getting sandbox: {e}")
            self._stats["failures"] += 1
            self._consecutive_failures += 1
            raise CodeExecutionError(f"Failed to get sandbox: {e}")        
        finally:
            if pooled_sb:
                should_recycle = (
                    not created_new and
                    self._should_recycle_sandbox(pooled_sb) and
                    self._running
                )
                if should_recycle:
                    # Double-check sandbox is alive and functional before returning to pool
                    if await self._is_sandbox_alive(pooled_sb.sandbox):
                        # Additional check: try a quick execution to ensure sandbox is fully functional
                        try:
                            await asyncio.wait_for(
                                asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda: pooled_sb.sandbox.exec("python", "-c", "import sys; print('ready')", timeout=2)
                                ),
                                timeout=3.0
                            )
                            
                            # Sandbox is healthy and functional - return to pool
                            try:
                                self._sandbox_queue.put_nowait(pooled_sb)
                                logger.debug("Returned healthy sandbox to pool")
                            except asyncio.QueueFull:
                                # Pool is full - terminate excess sandbox
                                await self._terminate_sandbox(pooled_sb.sandbox)
                                logger.debug("Pool full, terminated excess sandbox")
                        except Exception as e:
                            # Sandbox failed functional test - terminate it
                            logger.debug(f"Sandbox failed functional test, terminating: {e}")
                            await self._terminate_sandbox(pooled_sb.sandbox)
                    else:
                        # Sandbox is dead - terminate it
                        logger.debug("Sandbox is dead, terminating instead of recycling")
                        await self._terminate_sandbox(pooled_sb.sandbox)
                else:
                    # Should not recycle - terminate sandbox
                    await self._terminate_sandbox(pooled_sb.sandbox)
                    if not created_new:
                        self._stats["recycled"] += 1
                        logger.debug("Terminated sandbox (exceeded recycle criteria)")
    
    async def _create_sandbox(self) -> modal.Sandbox:
        """Create a new Modal sandbox with timeout protection."""
        try:
            # Add timeout protection for sandbox creation
            sandbox_creation = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: modal.Sandbox.create(
                    app=self.app,
                    image=self.image,
                    cpu=2.0,
                    memory=1024,
                    timeout=35
                )
            )
              # Wait for sandbox creation with timeout
            sandbox = await asyncio.wait_for(sandbox_creation, timeout=120)  # 2 minute timeout
            logger.debug(f"Created new sandbox of type: {type(sandbox)}")
            return sandbox
        except asyncio.TimeoutError:
            logger.error("Sandbox creation timed out after 2 minutes")
            raise Exception("Sandbox creation timed out - Modal may be experiencing issues")
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise
    
    async def _terminate_sandbox(self, sandbox: modal.Sandbox):
        """Safely terminate a sandbox with better error handling."""
        try:
            # Check if sandbox is still responsive before termination
            if hasattr(sandbox, '_terminated') and sandbox._terminated:
                logger.debug("Sandbox already terminated")
                return
                
            # Use asyncio timeout for termination
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, sandbox.terminate),
                timeout=10.0  # 10 second timeout for termination
            )
            logger.debug("Terminated sandbox successfully")
        except asyncio.TimeoutError:
            logger.warning("Sandbox termination timed out - may be unresponsive")
        except Exception as e:
            # Log the error but don't fail - sandbox may already be dead
            logger.warning(f"Failed to terminate sandbox (may already be dead): {e}")
            # Mark sandbox as terminated to avoid repeated attempts
            if hasattr(sandbox, '_terminated'):
                sandbox._terminated = True
    
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
        """Background task to maintain warm sandboxes in the pool with aggressive replenishment."""
        while self._running:
            try:
                current_size = self._sandbox_queue.qsize()
                
                # More aggressive warmup - start warming when below 90% capacity
                warmup_threshold = max(1, int(self.pool_size * 0.9))
                
                if current_size < warmup_threshold:
                    needed = self.pool_size - current_size
                    logger.info(f"Pool size ({current_size}) below threshold ({warmup_threshold}). Warming {needed} sandboxes...")
                    
                    # Create new sandboxes to fill the pool - but limit concurrent creation
                    max_concurrent = min(needed, 2)  # Don't overwhelm Modal
                    tasks = []
                    for _ in range(max_concurrent):
                        task = asyncio.create_task(self._create_and_queue_sandbox())
                        tasks.append(task)
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        # Log any failures
                        successful = 0
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                logger.warning(f"Failed to create sandbox {i+1}/{max_concurrent}: {result}")
                            else:
                                successful += 1
                        
                        if successful > 0:
                            logger.info(f"Successfully warmed {successful}/{max_concurrent} sandboxes")
                
                # Adaptive sleep interval based on pool health
                if current_size == 0:
                    # Critical: no sandboxes available
                    sleep_interval = 1
                elif current_size < warmup_threshold:
                    # Low: need more sandboxes
                    sleep_interval = 2
                else:
                    # Healthy: normal monitoring
                    sleep_interval = 5
                    
                await asyncio.sleep(sleep_interval)
                
            except Exception as e:
                logger.error(f"Error in warmup loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _create_and_queue_sandbox(self):
        """Create a sandbox and add it to the queue."""
        start_time = time.time()
        try:
            # Create the sandbox
            sandbox = await self._create_sandbox()
            creation_time = time.time() - start_time
            logger.info(f"Sandbox creation took {creation_time:.2f}s")
            
            # Proactively warm up the sandbox with core imports
            warmup_start = time.time()
            await self._warmup_sandbox_imports(sandbox)
            warmup_time = time.time() - warmup_start
            logger.info(f"Sandbox warmup with imports took {warmup_time:.2f}s")
            
            pooled_sb = PooledSandbox(
                sandbox=sandbox,
                created_at=time.time(),
                last_used=time.time()
            )
            
            try:
                self._sandbox_queue.put_nowait(pooled_sb)
                total_time = time.time() - start_time
                logger.info(f"Added warm sandbox to pool (total time: {total_time:.2f}s)")
            except asyncio.QueueFull:
                # Pool is full, terminate this sandbox
                await self._terminate_sandbox(sandbox)
                
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Failed to create and queue sandbox after {total_time:.2f}s: {e}")

    async def _warmup_sandbox_imports(self, sandbox: modal.Sandbox):
        """Warm up sandbox by importing core packages."""
        try:
            from mcp_hub.package_utils import get_warmup_import_commands
            
            # Get warmup commands
            import_commands = get_warmup_import_commands()
            warmup_script = "; ".join(import_commands)
            
            # Execute the warmup script
            logger.debug("Running sandbox warmup imports...")
            proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sandbox.exec("python", "-c", warmup_script, timeout=30)
            )
            
            # Check if warmup was successful
            if hasattr(proc, 'stdout') and hasattr(proc.stdout, 'read'):
                output = proc.stdout.read()
                if "Core packages warmed up successfully" in output:
                    logger.debug("Sandbox warmup imports completed successfully")
                else:
                    logger.warning(f"Sandbox warmup completed but output unexpected: {output}")
            else:
                logger.debug("Sandbox warmup imports completed")
                
        except Exception as e:
            logger.warning(f"Failed to warm up sandbox imports (sandbox still usable): {e}")
            
    async def _health_check_loop(self):
        """Background task to check sandbox health and perform proactive cleanup."""
        while self._running:
            try:
                # Perform regular health checks every interval
                await asyncio.sleep(self.health_check_interval)
                
                # First do a quick proactive cleanup
                cleaned = await self._proactive_cleanup()
                
                # Then do the full health check
                await self._perform_health_checks()
                
                # If we cleaned up sandboxes, trigger warmup
                if cleaned > 0:
                    logger.info(f"Health check cleaned {cleaned} sandboxes, pool may need warming")
                    
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
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
        """Check if a sandbox is alive by running a trivial command with better error handling."""
        try:
            # Check if sandbox was already marked as terminated
            if hasattr(sandbox, '_terminated') and sandbox._terminated:
                return False
                
            # Use a shorter timeout for liveness checks
            proc = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: sandbox.exec("python", "-c", "print('ping')", timeout=3)
                ),
                timeout=5.0  # Overall timeout
            )
            
            if hasattr(proc, "stdout") and hasattr(proc.stdout, "read"):
                out = proc.stdout.read()
                return "ping" in out
            else:
                # For some Modal versions, output might be returned directly
                out = str(proc)
                return "ping" in out
                
        except asyncio.TimeoutError:
            logger.debug("Liveness check timed out - sandbox likely dead")
            return False
        except Exception as e:
            logger.debug(f"Liveness check failed: {e}")
            # Mark sandbox as dead to avoid repeated checks
            if hasattr(sandbox, '_terminated'):
                sandbox._terminated = True
            return False
    
    async def _emergency_pool_reset(self):
        """Emergency reset of the pool when too many consecutive failures occur."""
        logger.warning("Performing emergency pool reset due to consecutive failures")
        
        # Drain and terminate all sandboxes in the pool
        terminated_count = 0
        while not self._sandbox_queue.empty():
            try:
                pooled_sb = self._sandbox_queue.get_nowait()
                await self._terminate_sandbox(pooled_sb.sandbox)
                terminated_count += 1
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Emergency reset: terminated {terminated_count} sandboxes")
        
        # Reset failure counter
        self._consecutive_failures = 0
        
        # Try to create one fresh sandbox to test if the underlying issue is resolved
        try:
            test_sandbox = await self._create_sandbox()
            test_pooled = PooledSandbox(
                sandbox=test_sandbox,
                created_at=time.time(),
                last_used=time.time(),
                use_count=0
            )            
            self._sandbox_queue.put_nowait(test_pooled)
            logger.info("Emergency reset successful: created test sandbox")
        except Exception as e:
            logger.error(f"Emergency reset failed to create test sandbox: {e}")
            # Still reset the counter to allow retries
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics including health metrics."""
        return {
            **self._stats,
            "pool_size": self._sandbox_queue.qsize(),
            "target_pool_size": self.pool_size,
            "running": self._running,
            "consecutive_failures": self._consecutive_failures,
            "last_successful_creation": self._last_successful_creation,
            "time_since_last_success": time.time() - self._last_successful_creation,
            "health_status": "healthy" if self._consecutive_failures < 3 else "degraded" if self._consecutive_failures < self._pool_reset_threshold else "critical"
        }

    async def _proactive_cleanup(self):
        """Proactively clean up dead or unhealthy sandboxes from the pool."""
        temp_sandboxes = []
        cleaned_count = 0
        
        # Drain the queue to check each sandbox
        while not self._sandbox_queue.empty():
            try:
                pooled_sb = self._sandbox_queue.get_nowait()
                
                # Quick health check
                if await self._is_sandbox_alive(pooled_sb.sandbox):
                    # Sandbox is alive - keep it
                    temp_sandboxes.append(pooled_sb)
                else:
                    # Sandbox is dead - terminate it
                    await self._terminate_sandbox(pooled_sb.sandbox)
                    cleaned_count += 1
                    logger.debug("Cleaned up dead sandbox during proactive cleanup")
                    
            except asyncio.QueueEmpty:
                break
        
        # Put healthy sandboxes back
        for pooled_sb in temp_sandboxes:
            try:
                self._sandbox_queue.put_nowait(pooled_sb)
            except asyncio.QueueFull:
                # Shouldn't happen, but terminate if it does
                await self._terminate_sandbox(pooled_sb.sandbox)
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Proactive cleanup removed {cleaned_count} dead sandboxes")
            
        return cleaned_count

# Helper function for testing and debugging the sandbox pool
async def test_sandbox_pool_health(pool: WarmSandboxPool) -> Dict[str, Any]:
    """Test sandbox pool health and return detailed diagnostics."""
    diagnostics: Dict[str, Any] = {
        "timestamp": time.time(),
        "pool_stats": pool.get_stats(),
        "tests": {}
    }
    
    logger.info("Starting sandbox pool health test...")
    
    # Test 1: Pool basic stats
    stats = pool.get_stats()
    diagnostics["tests"]["pool_stats"] = {
        "passed": True,
        "details": stats
    }
    
    # Test 2: Try to get a sandbox
    try:
        async with pool.get_sandbox(timeout=10.0) as sandbox:
            # Test 3: Try to run a simple command
            try:
                proc = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: sandbox.exec("python", "-c", "print('health_test_ok')", timeout=5)
                )
                output = proc.stdout.read() if hasattr(proc.stdout, "read") else str(proc)
                
                diagnostics["tests"]["sandbox_execution"] = {
                    "passed": "health_test_ok" in output,
                    "output": output[:200],  # First 200 chars
                    "details": "Successfully executed test command"
                }
            except Exception as e:
                diagnostics["tests"]["sandbox_execution"] = {
                    "passed": False,
                    "error": str(e),
                    "details": "Failed to execute test command in sandbox"
                }
        
        diagnostics["tests"]["sandbox_acquisition"] = {
            "passed": True,
            "details": "Successfully acquired and released sandbox"
        }
        
    except Exception as e:
        diagnostics["tests"]["sandbox_acquisition"] = {
            "passed": False,
            "error": str(e),
            "details": "Failed to acquire sandbox from pool"
        }
        
        diagnostics["tests"]["sandbox_execution"] = {
            "passed": False,
            "error": "Could not test - no sandbox available",
            "details": "Skipped due to sandbox acquisition failure"
        }
    
    # Test 4: Check pool warmup status
    if pool._running:
        warmup_needed = pool.pool_size - stats["pool_size"]
        diagnostics["tests"]["pool_warmup"] = {
            "passed": warmup_needed <= 1,  # Allow 1 sandbox to be missing
            "details": f"Pool has {stats['pool_size']}/{pool.pool_size} sandboxes, {warmup_needed} needed"
        }
    else:
        diagnostics["tests"]["pool_warmup"] = {
            "passed": False,
            "details": "Pool is not running"
        }
    
    # Overall health assessment
    all_tests_passed = all(test.get("passed", False) for test in diagnostics["tests"].values())
    diagnostics["overall_health"] = "healthy" if all_tests_passed else "unhealthy"
    
    logger.info(f"Sandbox pool health test completed. Overall health: {diagnostics['overall_health']}")
    return diagnostics
