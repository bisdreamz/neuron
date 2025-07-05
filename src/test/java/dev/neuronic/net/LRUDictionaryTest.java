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
        LRUDictionary dict = new LRUDictionary(5, 100);
        
        // Add some values - get distributed indices
        int firstIndex = dict.getIndex("first");
        int secondIndex = dict.getIndex("second");
        int intIndex = dict.getIndex(123);
        int floatIndex = dict.getIndex(45.6f);
        
        // Verify indices are in valid range (0 to maxSize-1)
        assertTrue(firstIndex >= 0 && firstIndex < 5);
        assertTrue(secondIndex >= 0 && secondIndex < 5);
        assertTrue(intIndex >= 0 && intIndex < 5);
        assertTrue(floatIndex >= 0 && floatIndex < 5);
        
        // Verify retrieval
        assertEquals("first", dict.getValue(firstIndex));
        assertEquals("second", dict.getValue(secondIndex));
        assertEquals(123, dict.getValue(intIndex));
        assertEquals(45.6f, dict.getValue(floatIndex));
        
        // Verify size
        assertEquals(4, dict.size());
        assertEquals(5, dict.getMaxSize());
    }
    
    @Test
    void testLRUEviction() {
        LRUDictionary dict = new LRUDictionary(3, 100);
        
        // Fill to capacity - store indices
        int indexA = dict.getIndex("A");
        int indexB = dict.getIndex("B");
        int indexC = dict.getIndex("C");
        assertEquals(3, dict.size());
        
        // Add one more - should evict LRU entry
        int indexD = dict.getIndex("D");
        assertEquals(3, dict.size()); // Size stays at max
        
        // Verify D is present with valid index
        assertTrue(indexD >= 0 && indexD < 3);
        assertEquals("D", dict.getValue(indexD));
        
        // Should have evicted one item (likely A as oldest)
        // Check that not all original items are still present
        int remainingCount = 0;
        if (dict.containsValue("A")) remainingCount++;
        if (dict.containsValue("B")) remainingCount++;
        if (dict.containsValue("C")) remainingCount++;
        
        assertEquals(2, remainingCount); // Only 2 of original 3 should remain
        assertTrue(dict.containsValue("D")); // D should definitely be present
    }
    
    @Test
    void testAccessOrderUpdate() {
        LRUDictionary dict = new LRUDictionary(3, 100);
        
        // Add A, B, C
        int indexA = dict.getIndex("A");
        dict.getIndex("B");
        dict.getIndex("C");
        
        // Access A again - moves it to most recent, should return same index
        assertEquals(indexA, dict.getIndex("A"));
        
        // Add D - should evict oldest (after A was refreshed)
        dict.getIndex("D");
        
        // A should still be there (refreshed), and D should be there
        assertTrue(dict.containsValue("A"));
        assertTrue(dict.containsValue("D"));
        
        // One of B or C should be evicted
        int remainingOriginals = 0;
        if (dict.containsValue("B")) remainingOriginals++;
        if (dict.containsValue("C")) remainingOriginals++;
        
        assertEquals(1, remainingOriginals); // One of B or C should remain
        assertEquals(3, dict.size());
    }
    
    @Test
    void testMaxSizeConstraint() {
        LRUDictionary dict = new LRUDictionary(10, 100);
        
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
        LRUDictionary dict = new LRUDictionary(5, 100);
        assertThrows(IllegalArgumentException.class, () -> dict.getIndex(null));
    }
    
    @Test
    void testInvalidMaxSize() {
        assertThrows(IllegalArgumentException.class, () -> new LRUDictionary(0, 100));
        assertThrows(IllegalArgumentException.class, () -> new LRUDictionary(-1, 100));
    }
    
    @Test
    void testGetValue() {
        LRUDictionary dict = new LRUDictionary(5, 100);
        
        int testIndex = dict.getIndex("test");
        assertEquals("test", dict.getValue(testIndex));
        
        // Non-existent index
        assertNull(dict.getValue(999));
    }
    
    @Test
    void testContains() {
        LRUDictionary dict = new LRUDictionary(5, 100);
        
        int existsIndex = dict.getIndex("exists");
        
        assertTrue(dict.containsValue("exists"));
        assertFalse(dict.containsValue("not-exists"));
        
        assertTrue(dict.containsIndex(existsIndex));
        assertFalse(dict.containsIndex(999));
    }
    
    @Test
    void testClear() {
        LRUDictionary dict = new LRUDictionary(5, 100);
        
        dict.getIndex("A");
        dict.getIndex("B");
        dict.getIndex("C");
        
        assertEquals(3, dict.size());
        
        dict.clear();
        
        assertEquals(0, dict.size());
        assertFalse(dict.containsValue("A"));
        assertFalse(dict.containsIndex(0));
        
        // Can add new items after clear - get distributed index
        int newIndex = dict.getIndex("D");
        assertTrue(newIndex >= 0 && newIndex < 5);
    }
    
    @Test
    void testSerialization() throws IOException {
        LRUDictionary dict = new LRUDictionary(5, 100);
        
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
        LRUDictionary dict2 = LRUDictionary.readFrom(in, 100);
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
        LRUDictionary dict = new LRUDictionary(5, 100);
        
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
        LRUDictionary dict = new LRUDictionary(5, 100);
        dict.getIndex("A");
        dict.getIndex("B");
        
        String str = dict.toString();
        assertTrue(str.contains("LRUDictionary"));
        assertTrue(str.contains("size=2/5"));
    }
    
    @Test
    void testConcurrentAccess() throws InterruptedException {
        final int maxSize = 100;
        final LRUDictionary dict = new LRUDictionary(maxSize, 1000);
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
        LRUDictionary dict = new LRUDictionary(3, 100);
        
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