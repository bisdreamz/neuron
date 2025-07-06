package dev.neuronic.net.common;

import org.junit.jupiter.api.Test;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for PooledFloatArray3D buffer pooling functionality.
 */
class PooledFloatArray3DTest {
    
    @Test
    void testBasicAcquireRelease() {
        PooledFloatArray3D pool = new PooledFloatArray3D(2, 3, 4);
        
        // Get buffer
        float[][][] buffer = pool.getBuffer();
        assertNotNull(buffer);
        assertEquals(2, buffer.length);
        assertEquals(3, buffer[0].length);
        assertEquals(4, buffer[0][0].length);
        
        // Verify zeroed
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 4; k++) {
                    assertEquals(0.0f, buffer[i][j][k]);
                }
            }
        }
        
        // Modify and release
        buffer[0][0][0] = 1.0f;
        pool.releaseBuffer(buffer);
        
        // Get again - should be zeroed
        float[][][] buffer2 = pool.getBuffer();
        assertEquals(0.0f, buffer2[0][0][0]);
        
        // Should reuse same buffer
        assertSame(buffer, buffer2);
    }
    
    @Test
    void testGetBufferWithoutZeroing() {
        PooledFloatArray3D pool = new PooledFloatArray3D(1, 2, 3);
        
        // Get buffer and modify
        float[][][] buffer = pool.getBuffer();
        buffer[0][0][0] = 5.0f;
        pool.releaseBuffer(buffer);
        
        // Get without zeroing
        float[][][] buffer2 = pool.getBuffer(false);
        assertEquals(5.0f, buffer2[0][0][0]); // Should still have old value
    }
    
    @Test
    void testDimensionValidation() {
        assertThrows(IllegalArgumentException.class, () -> new PooledFloatArray3D(0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new PooledFloatArray3D(1, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> new PooledFloatArray3D(1, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new PooledFloatArray3D(-1, 2, 3));
    }
    
    @Test
    void testReleaseWrongSizeBuffer() {
        PooledFloatArray3D pool = new PooledFloatArray3D(2, 3, 4);
        
        // Try to release null - should be safe
        pool.releaseBuffer(null);
        
        // Try to release wrong size - should be ignored
        float[][][] wrongSize = new float[1][3][4];
        pool.releaseBuffer(wrongSize);
        assertEquals(0, pool.getPoolSize());
        
        // Release correct size
        float[][][] correctSize = new float[2][3][4];
        pool.releaseBuffer(correctSize);
        assertEquals(1, pool.getPoolSize());
    }
    
    @Test
    void testGetDimensions() {
        PooledFloatArray3D pool = new PooledFloatArray3D(5, 10, 15);
        int[] dims = pool.getDimensions();
        assertArrayEquals(new int[]{5, 10, 15}, dims);
    }
    
    @Test
    void testPoolGrowth() {
        PooledFloatArray3D pool = new PooledFloatArray3D(2, 3, 4);
        assertEquals(0, pool.getPoolSize());
        
        // Get multiple buffers
        float[][][] buffer1 = pool.getBuffer();
        float[][][] buffer2 = pool.getBuffer();
        float[][][] buffer3 = pool.getBuffer();
        
        assertNotSame(buffer1, buffer2);
        assertNotSame(buffer2, buffer3);
        
        // Release them
        pool.releaseBuffer(buffer1);
        assertEquals(1, pool.getPoolSize());
        pool.releaseBuffer(buffer2);
        assertEquals(2, pool.getPoolSize());
        pool.releaseBuffer(buffer3);
        assertEquals(3, pool.getPoolSize());
    }
    
    @Test
    void testConcurrentAccess() throws InterruptedException {
        final int numThreads = 10;
        final int iterations = 1000;
        final PooledFloatArray3D pool = new PooledFloatArray3D(3, 4, 5);
        final CountDownLatch startLatch = new CountDownLatch(1);
        final CountDownLatch endLatch = new CountDownLatch(numThreads);
        final AtomicInteger successCount = new AtomicInteger(0);
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        for (int t = 0; t < numThreads; t++) {
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < iterations; i++) {
                        float[][][] buffer = pool.getBuffer();
                        
                        // Verify dimensions
                        if (buffer.length == 3 && buffer[0].length == 4 && buffer[0][0].length == 5) {
                            // Verify zeroed
                            boolean allZero = true;
                            for (int a = 0; a < 3 && allZero; a++) {
                                for (int b = 0; b < 4 && allZero; b++) {
                                    for (int c = 0; c < 5 && allZero; c++) {
                                        if (buffer[a][b][c] != 0.0f) {
                                            allZero = false;
                                        }
                                    }
                                }
                            }
                            
                            if (allZero) {
                                // Modify buffer
                                buffer[0][0][0] = Thread.currentThread().getId();
                                
                                // Small delay to increase contention
                                Thread.yield();
                                
                                pool.releaseBuffer(buffer);
                                successCount.incrementAndGet();
                            }
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    endLatch.countDown();
                }
            });
        }
        
        // Start all threads
        startLatch.countDown();
        
        // Wait for completion
        assertTrue(endLatch.await(10, TimeUnit.SECONDS));
        executor.shutdown();
        
        // All operations should succeed
        assertEquals(numThreads * iterations, successCount.get());
        
        // Pool should have some buffers (but not necessarily all due to concurrency)
        assertTrue(pool.getPoolSize() > 0);
    }
    
    @Test
    void testMemoryEfficiency() {
        PooledFloatArray3D pool = new PooledFloatArray3D(10, 20, 30);
        
        // Track unique buffer instances
        float[][][] buffer1 = pool.getBuffer();
        pool.releaseBuffer(buffer1);
        
        float[][][] buffer2 = pool.getBuffer();
        assertSame(buffer1, buffer2); // Should reuse
        
        float[][][] buffer3 = pool.getBuffer(); // New allocation
        assertNotSame(buffer2, buffer3);
        
        pool.releaseBuffer(buffer2);
        pool.releaseBuffer(buffer3);
        
        // Now pool has 2 buffers
        assertEquals(2, pool.getPoolSize());
    }
    
    @Test
    void testZeroBuffer() {
        PooledFloatArray3D pool = new PooledFloatArray3D(2, 2, 2);
        
        // Get buffer and modify all elements
        float[][][] buffer = pool.getBuffer();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    buffer[i][j][k] = i + j + k;
                }
            }
        }
        
        // Release and get again - should be zeroed
        pool.releaseBuffer(buffer);
        float[][][] buffer2 = pool.getBuffer();
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    assertEquals(0.0f, buffer2[i][j][k]);
                }
            }
        }
    }
}