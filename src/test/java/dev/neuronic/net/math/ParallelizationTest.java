package dev.neuronic.net.math;

import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test parallelization utilities and heuristics.
 */
class ParallelizationTest {

    @Test
    void testShouldParallelizeHeuristics() {
        ExecutorService executor = Executors.newFixedThreadPool(4);
        
        try {
            // Small work - should not parallelize
            assertFalse(Parallelization.shouldParallelize(100, executor));
            assertFalse(Parallelization.shouldParallelize(1000, executor));
            
            // Large work - should parallelize (default threshold is 1024 * 2 = 2048)
            assertTrue(Parallelization.shouldParallelize(3000, executor));
            assertTrue(Parallelization.shouldParallelize(10000, executor));
            
            // No executor - should never parallelize
            assertFalse(Parallelization.shouldParallelize(10000, null));
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testCalculateOptimalThreads() {
        ExecutorService executor = Executors.newFixedThreadPool(4);
        
        try {
            // Small work - should use 1 thread
            assertEquals(1, Parallelization.calculateOptimalThreads(1000, executor));
            
            // Medium work - should use 2 threads (2048 work / 1024 per thread)
            assertEquals(2, Parallelization.calculateOptimalThreads(2048, executor));
            
            // Large work - should use multiple threads but not exceed max
            int threads = Parallelization.calculateOptimalThreads(10000, executor);
            assertTrue(threads > 1);
            assertTrue(threads <= Parallelization.getMaxThreadsPerTask());
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testSplitWork() {
        // Test even split
        Parallelization.WorkRange[] ranges = Parallelization.splitWork(100, 4);
        assertEquals(4, ranges.length);
        assertEquals(25, ranges[0].size);
        assertEquals(25, ranges[1].size);
        assertEquals(25, ranges[2].size);
        assertEquals(25, ranges[3].size);
        
        // Verify ranges cover full work
        assertEquals(0, ranges[0].start);
        assertEquals(25, ranges[0].end);
        assertEquals(25, ranges[1].start);
        assertEquals(50, ranges[1].end);
        assertEquals(75, ranges[3].start);
        assertEquals(100, ranges[3].end);
        
        // Test uneven split
        ranges = Parallelization.splitWork(101, 4);
        assertEquals(4, ranges.length);
        assertEquals(26, ranges[0].size); // Gets the extra work
        assertEquals(25, ranges[1].size);
        assertEquals(25, ranges[2].size);
        assertEquals(25, ranges[3].size);
        
        // Single thread
        ranges = Parallelization.splitWork(100, 1);
        assertEquals(1, ranges.length);
        assertEquals(0, ranges[0].start);
        assertEquals(100, ranges[0].end);
    }
    
    @Test
    void testExecuteParallel() {
        ExecutorService executor = Executors.newFixedThreadPool(4);
        AtomicInteger counter = new AtomicInteger(0);
        
        try {
            Runnable task1 = () -> counter.addAndGet(1);
            Runnable task2 = () -> counter.addAndGet(10);
            Runnable task3 = () -> counter.addAndGet(100);
            
            Parallelization.executeParallel(executor, task1, task2, task3);
            
            // All tasks should have completed
            assertEquals(111, counter.get());
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testConfigurableThresholds() {
        int originalThreshold = Parallelization.getMinWorkSizePerThread();
        int originalMaxThreads = Parallelization.getMaxThreadsPerTask();
        int originalThreads = Parallelization.getThreads();
        
        try {
            // Test setting thresholds
            Parallelization.setMinWorkSizePerThread(500);
            assertEquals(500, Parallelization.getMinWorkSizePerThread());
            
            Parallelization.setMaxThreadsPerTask(8);
            assertEquals(8, Parallelization.getMaxThreadsPerTask());
            
            Parallelization.setThreads(6);
            assertEquals(6, Parallelization.getThreads());
            
            // Test invalid values
            assertThrows(IllegalArgumentException.class, 
                        () -> Parallelization.setMinWorkSizePerThread(0));
            assertThrows(IllegalArgumentException.class, 
                        () -> Parallelization.setMaxThreadsPerTask(-1));
            assertThrows(IllegalArgumentException.class, 
                        () -> Parallelization.setThreads(0));
            
        } finally {
            // Restore original values
            Parallelization.setMinWorkSizePerThread(originalThreshold);
            Parallelization.setMaxThreadsPerTask(originalMaxThreads);
            Parallelization.setThreads(originalThreads);
        }
    }
}