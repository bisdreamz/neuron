package dev.neuronic.net.math;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.CountDownLatch;
import java.util.ArrayList;
import java.util.List;

/**
 * Central configuration and utilities for thread-based parallelization.
 * Provides shared heuristics for when to parallelize work and utilities for splitting tasks.
 */
public final class Parallelization {
    
    // Configuration - adjustable based on hardware and workload
    private static int MIN_WORK_SIZE_PER_THREAD = 1024; // Minimum work per thread to be worthwhile
    private static int MAX_THREADS_PER_TASK = Runtime.getRuntime().availableProcessors();
    private static int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();
    
    /**
     * Check if the given work size should be parallelized across threads.
     * 
     * @param workSize total amount of work (e.g., array length, number of operations)
     * @param executor the executor service (null means no parallelization possible)
     * @return true if work should be split across multiple threads
     */
    public static boolean shouldParallelize(int workSize, ExecutorService executor) {
        return executor != null && 
               workSize >= MIN_WORK_SIZE_PER_THREAD * 2; // At least 2 threads worth of work
    }
    
    /**
     * Calculate optimal number of threads for the given work size.
     * 
     * @param workSize total amount of work
     * @param executor the executor service
     * @return number of threads to use (1 = don't parallelize)
     */
    public static int calculateOptimalThreads(int workSize, ExecutorService executor) {
        if (!shouldParallelize(workSize, executor))
            return 1;
        
        // Calculate how many threads we can effectively use
        int maxThreadsForWork = workSize / MIN_WORK_SIZE_PER_THREAD;
        return Math.min(Math.min(maxThreadsForWork, MAX_THREADS_PER_TASK), DEFAULT_THREAD_COUNT);
    }
    
    /**
     * Split work evenly across threads.
     * 
     * @param totalWork total work size
     * @param numThreads number of threads to split across
     * @return array of WorkRange objects defining start/end for each thread
     */
    public static WorkRange[] splitWork(int totalWork, int numThreads) {
        if (numThreads <= 1) {
            return new WorkRange[]{new WorkRange(0, totalWork)};
        }
        
        WorkRange[] ranges = new WorkRange[numThreads];
        int workPerThread = totalWork / numThreads;
        int remainder = totalWork % numThreads;
        
        int start = 0;
        for (int i = 0; i < numThreads; i++) {
            int size = workPerThread + (i < remainder ? 1 : 0);
            ranges[i] = new WorkRange(start, start + size);
            start += size;
        }
        
        return ranges;
    }
    
    /**
     * Execute parallel work using CountDownLatch for synchronization.
     * This is more efficient than Future.get() for fire-and-forget parallel tasks.
     * 
     * @param executor the executor service
     * @param tasks the tasks to execute in parallel
     */
    public static void executeParallel(ExecutorService executor, Runnable... tasks) {
        if (tasks.length <= 1) {
            // Just run directly if only one task
            for (Runnable task : tasks) {
                task.run();
            }
            return;
        }
        
        CountDownLatch latch = new CountDownLatch(tasks.length);
        
        for (Runnable task : tasks) {
            executor.submit(() -> {
                try {
                    task.run();
                } finally {
                    latch.countDown();
                }
            });
        }
        
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Parallel execution interrupted", e);
        }
    }
    
    /**
     * Get the minimum work size per thread threshold.
     * @return current threshold
     */
    public static int getMinWorkSizePerThread() {
        return MIN_WORK_SIZE_PER_THREAD;
    }
    
    /**
     * Set the minimum work size per thread threshold.
     * Lower values = more aggressive parallelization, higher overhead.
     * Higher values = less parallelization, lower overhead.
     * 
     * @param minWorkSize new threshold (must be positive)
     */
    public static void setMinWorkSizePerThread(int minWorkSize) {
        if (minWorkSize <= 0)
            throw new IllegalArgumentException("Min work size must be positive: " + minWorkSize);
        MIN_WORK_SIZE_PER_THREAD = minWorkSize;
    }
    
    /**
     * Get the maximum threads per task.
     * @return current max threads
     */
    public static int getMaxThreadsPerTask() {
        return MAX_THREADS_PER_TASK;
    }
    
    /**
     * Set the maximum threads per task.
     * @param maxThreads new max threads (must be positive)
     */
    public static void setMaxThreadsPerTask(int maxThreads) {
        if (maxThreads <= 0)
            throw new IllegalArgumentException("Max threads must be positive: " + maxThreads);
        MAX_THREADS_PER_TASK = maxThreads;
    }
    
    /**
     * Get the default thread count used for parallelization.
     * @return current default thread count
     */
    public static int getThreads() {
        return DEFAULT_THREAD_COUNT;
    }
    
    /**
     * Set the default thread count for parallelization.
     * This affects how many threads will be used when splitting parallel work.
     * 
     * @param threads new thread count (must be positive)
     */
    public static void setThreads(int threads) {
        if (threads <= 0)
            throw new IllegalArgumentException("Thread count must be positive: " + threads);
        DEFAULT_THREAD_COUNT = threads;
    }
    
    /**
     * Represents a range of work for a single thread.
     */
    public static class WorkRange {
        public final int start;
        public final int end;
        public final int size;
        
        public WorkRange(int start, int end) {
            this.start = start;
            this.end = end;
            this.size = end - start;
        }
        
        @Override
        public String toString() {
            return String.format("WorkRange[%d-%d, size=%d]", start, end, size);
        }
    }
    
    private Parallelization() {} // Prevent instantiation
}