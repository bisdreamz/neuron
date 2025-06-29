package dev.neuronic.net;

import org.junit.jupiter.api.Test;
import java.io.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for LRUDictionary with eviction behavior.
 */
class LRUDictionaryTest {
    
    @Test
    void testBasicOperations() {
        LRUDictionary dict = new LRUDictionary(5);
        
        // Add some values
        assertEquals(0, dict.getIndex("first"));
        assertEquals(1, dict.getIndex("second"));
        assertEquals(2, dict.getIndex(123));
        assertEquals(3, dict.getIndex(45.6f));
        
        // Verify retrieval
        assertEquals("first", dict.getValue(0));
        assertEquals("second", dict.getValue(1));
        assertEquals(123, dict.getValue(2));
        assertEquals(45.6f, dict.getValue(3));
        
        // Verify size
        assertEquals(4, dict.size());
        assertEquals(5, dict.getMaxSize());
    }
    
    @Test
    void testLRUEviction() {
        LRUDictionary dict = new LRUDictionary(3);
        
        // Fill to capacity
        assertEquals(0, dict.getIndex("A"));
        assertEquals(1, dict.getIndex("B"));
        assertEquals(2, dict.getIndex("C"));
        assertEquals(3, dict.size());
        
        // Add one more - should evict "A" (oldest) and reuse its index
        assertEquals(0, dict.getIndex("D")); // Reuses index 0 from evicted "A"
        assertEquals(3, dict.size()); // Size stays at max
        
        // "A" should be gone, replaced by "D"
        assertEquals("D", dict.getValue(0));
        assertFalse(dict.containsValue("A"));
        
        // B, C should still be there
        assertEquals("B", dict.getValue(1));
        assertEquals("C", dict.getValue(2));
    }
    
    @Test
    void testAccessOrderUpdate() {
        LRUDictionary dict = new LRUDictionary(3);
        
        // Add A, B, C
        dict.getIndex("A");
        dict.getIndex("B");
        dict.getIndex("C");
        
        // Access A again - moves it to most recent
        assertEquals(0, dict.getIndex("A"));
        
        // Add D - should evict B (now oldest)
        dict.getIndex("D");
        
        // Verify B is gone, A is still there
        assertFalse(dict.containsValue("B"));
        assertTrue(dict.containsValue("A"));
        assertTrue(dict.containsValue("C"));
        assertTrue(dict.containsValue("D"));
    }
    
    @Test
    void testMaxSizeConstraint() {
        LRUDictionary dict = new LRUDictionary(10);
        
        // Add 15 items
        for (int i = 0; i < 15; i++) {
            dict.getIndex("item" + i);
        }
        
        // Size should be capped at 10
        assertEquals(10, dict.size());
        
        // First 5 should be evicted
        for (int i = 0; i < 5; i++) {
            assertFalse(dict.containsValue("item" + i));
        }
        
        // Last 10 should be present
        for (int i = 5; i < 15; i++) {
            assertTrue(dict.containsValue("item" + i));
        }
    }
    
    @Test
    void testNullValue() {
        LRUDictionary dict = new LRUDictionary(5);
        assertThrows(IllegalArgumentException.class, () -> dict.getIndex(null));
    }
    
    @Test
    void testInvalidMaxSize() {
        assertThrows(IllegalArgumentException.class, () -> new LRUDictionary(0));
        assertThrows(IllegalArgumentException.class, () -> new LRUDictionary(-1));
    }
    
    @Test
    void testGetValue() {
        LRUDictionary dict = new LRUDictionary(5);
        
        dict.getIndex("test");
        assertEquals("test", dict.getValue(0));
        
        // Non-existent index
        assertNull(dict.getValue(999));
    }
    
    @Test
    void testContains() {
        LRUDictionary dict = new LRUDictionary(5);
        
        dict.getIndex("exists");
        
        assertTrue(dict.containsValue("exists"));
        assertFalse(dict.containsValue("not-exists"));
        
        assertTrue(dict.containsIndex(0));
        assertFalse(dict.containsIndex(999));
    }
    
    @Test
    void testClear() {
        LRUDictionary dict = new LRUDictionary(5);
        
        dict.getIndex("A");
        dict.getIndex("B");
        dict.getIndex("C");
        
        assertEquals(3, dict.size());
        
        dict.clear();
        
        assertEquals(0, dict.size());
        assertFalse(dict.containsValue("A"));
        assertFalse(dict.containsIndex(0));
        
        // Can add new items after clear
        assertEquals(0, dict.getIndex("D"));
    }
    
    @Test
    void testSerialization() throws IOException {
        LRUDictionary dict = new LRUDictionary(5);
        
        // Add mixed types
        dict.getIndex("string");
        dict.getIndex(42);
        dict.getIndex(3.14f);
        
        // Serialize
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(baos);
        dict.writeTo(out);
        out.close();
        
        // Deserialize
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        DataInputStream in = new DataInputStream(bais);
        LRUDictionary dict2 = LRUDictionary.readFrom(in);
        in.close();
        
        // Verify
        assertEquals(5, dict2.getMaxSize());
        assertEquals(3, dict2.size());
        assertTrue(dict2.containsValue("string"));
        assertTrue(dict2.containsValue(42));
        assertTrue(dict2.containsValue(3.14f));
    }
    
    @Test
    void testGetNextIndex() {
        LRUDictionary dict = new LRUDictionary(5);
        
        assertEquals(0, dict.getNextIndex());
        dict.getIndex("A");
        assertEquals(1, dict.getNextIndex());
        dict.getIndex("B");
        assertEquals(2, dict.getNextIndex());
        
        // Fill to capacity
        dict.getIndex("C");
        dict.getIndex("D");
        dict.getIndex("E");
        assertEquals(5, dict.getNextIndex()); // At capacity
        
        // Adding more doesn't increase next index
        dict.getIndex("F"); // Evicts "A"
        assertEquals(5, dict.getNextIndex()); // Still at capacity
    }
    
    @Test
    void testToString() {
        LRUDictionary dict = new LRUDictionary(5);
        dict.getIndex("A");
        dict.getIndex("B");
        
        String str = dict.toString();
        assertTrue(str.contains("LRUDictionary"));
        assertTrue(str.contains("size=2/5"));
    }
    
    @Test
    void testConcurrentAccess() throws InterruptedException {
        final int maxSize = 100;
        final LRUDictionary dict = new LRUDictionary(maxSize);
        final int numThreads = 10;
        final int itemsPerThread = 200;
        final CountDownLatch startLatch = new CountDownLatch(1);
        final CountDownLatch endLatch = new CountDownLatch(numThreads);
        final AtomicInteger successCount = new AtomicInteger(0);
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    
                    for (int i = 0; i < itemsPerThread; i++) {
                        String key = "thread" + threadId + "_item" + i;
                        int index = dict.getIndex(key);
                        
                        // Verify we can retrieve it
                        if (dict.containsValue(key)) {
                            successCount.incrementAndGet();
                        }
                        
                        // Access some existing items to change LRU order
                        if (i % 10 == 0 && i > 0) {
                            dict.getIndex("thread" + threadId + "_item" + (i - 10));
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
        
        // Dictionary size should be at max
        assertEquals(maxSize, dict.size());
        
        // Should have successful operations
        assertTrue(successCount.get() > 0);
    }
    
    @Test
    void testEvictionOrder() {
        LRUDictionary dict = new LRUDictionary(3);
        
        // Add A, B, C
        dict.getIndex("A");
        dict.getIndex("B");
        dict.getIndex("C");
        
        // Access in order B, A, C (making B least recently used of the three)
        dict.containsValue("B");
        dict.containsValue("A");
        dict.containsValue("C");
        
        // Add D - should evict A (least recently accessed via getIndex)
        dict.getIndex("D");
        
        // A should be evicted (it was accessed via containsValue, not getIndex)
        assertFalse(dict.containsValue("A"));
        assertTrue(dict.containsValue("B"));
        assertTrue(dict.containsValue("C"));
        assertTrue(dict.containsValue("D"));
    }
}