package dev.neuronic.net;

import org.junit.jupiter.api.Test;
import java.io.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for basic Dictionary functionality.
 */
class DictionaryTest {
    
    @Test
    void testBasicOperations() {
        Dictionary dict = new Dictionary();
        
        // First value gets index 0
        assertEquals(0, dict.getIndex("first"));
        assertEquals(1, dict.getIndex("second"));
        assertEquals(2, dict.getIndex(123));
        assertEquals(3, dict.getIndex(45.6f));
        
        // Repeated values return same index
        assertEquals(0, dict.getIndex("first"));
        assertEquals(2, dict.getIndex(123));
        
        // Check size
        assertEquals(4, dict.size());
        assertEquals(4, dict.getNextIndex());
    }
    
    @Test
    void testGetValue() {
        Dictionary dict = new Dictionary();
        
        dict.getIndex("test");
        dict.getIndex(42);
        
        assertEquals("test", dict.getValue(0));
        assertEquals(42, dict.getValue(1));
        assertNull(dict.getValue(999)); // Non-existent
    }
    
    @Test
    void testContains() {
        Dictionary dict = new Dictionary();
        
        dict.getIndex("exists");
        
        assertTrue(dict.containsValue("exists"));
        assertFalse(dict.containsValue("not-exists"));
        
        assertTrue(dict.containsIndex(0));
        assertFalse(dict.containsIndex(1));
    }
    
    @Test
    void testNullValue() {
        Dictionary dict = new Dictionary();
        assertThrows(IllegalArgumentException.class, () -> dict.getIndex(null));
    }
    
    @Test
    void testClear() {
        Dictionary dict = new Dictionary();
        
        dict.getIndex("A");
        dict.getIndex("B");
        dict.getIndex("C");
        
        assertEquals(3, dict.size());
        
        dict.clear();
        
        assertEquals(0, dict.size());
        assertEquals(0, dict.getNextIndex());
        assertFalse(dict.containsValue("A"));
        assertFalse(dict.containsIndex(0));
        
        // Can add after clear
        assertEquals(0, dict.getIndex("D"));
    }
    
    @Test
    void testMixedTypes() {
        Dictionary dict = new Dictionary();
        
        // Different types
        assertEquals(0, dict.getIndex("string"));
        assertEquals(1, dict.getIndex(42));
        assertEquals(2, dict.getIndex(3.14f));
        assertEquals(3, dict.getIndex(true));
        assertEquals(4, dict.getIndex(99L));
        
        // Verify retrieval
        assertEquals("string", dict.getValue(0));
        assertEquals(42, dict.getValue(1));
        assertEquals(3.14f, dict.getValue(2));
        assertEquals(true, dict.getValue(3));
        assertEquals(99L, dict.getValue(4));
    }
    
    @Test
    void testSerialization() throws IOException {
        Dictionary dict = new Dictionary();
        
        // Add various types
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
        Dictionary dict2 = Dictionary.readFrom(in);
        in.close();
        
        // Verify
        assertEquals(3, dict2.size());
        assertEquals(3, dict2.getNextIndex());
        
        // Values preserved (as strings due to serialization)
        assertTrue(dict2.containsValue("string"));
        assertTrue(dict2.containsValue(42)); // Parsed back as Integer
        assertTrue(dict2.containsValue(3.14f)); // Parsed back as Float
    }
    
    @Test
    void testGetSerializedSize() {
        Dictionary dict = new Dictionary();
        
        int initialSize = dict.getSerializedSize();
        assertTrue(initialSize > 0);
        
        dict.getIndex("test");
        int afterOneSize = dict.getSerializedSize();
        assertTrue(afterOneSize > initialSize);
        
        dict.getIndex("longer string value");
        int afterTwoSize = dict.getSerializedSize();
        assertTrue(afterTwoSize > afterOneSize);
    }
    
    @Test
    void testToString() {
        Dictionary dict = new Dictionary();
        dict.getIndex("A");
        dict.getIndex("B");
        
        String str = dict.toString();
        assertTrue(str.contains("Dictionary"));
        assertTrue(str.contains("size=2"));
        assertTrue(str.contains("nextIndex=2"));
    }
    
    @Test
    void testConcurrentAccess() throws InterruptedException {
        final Dictionary dict = new Dictionary();
        final int numThreads = 10;
        final int itemsPerThread = 1000;
        final CountDownLatch startLatch = new CountDownLatch(1);
        final CountDownLatch endLatch = new CountDownLatch(numThreads);
        final ConcurrentHashMap<String, Integer> expectedIndices = new ConcurrentHashMap<>();
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    
                    for (int i = 0; i < itemsPerThread; i++) {
                        String key = "thread" + threadId + "_item" + i;
                        int index = dict.getIndex(key);
                        
                        // Store first occurrence
                        expectedIndices.putIfAbsent(key, index);
                        
                        // Verify consistency
                        assertEquals(expectedIndices.get(key).intValue(), index);
                        
                        // Also test retrieval
                        assertEquals(key, dict.getValue(index));
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
        
        // Verify total size
        assertEquals(numThreads * itemsPerThread, dict.size());
        
        // Verify all mappings are consistent
        for (var entry : expectedIndices.entrySet()) {
            assertEquals(entry.getValue().intValue(), dict.getIndex(entry.getKey()));
        }
    }
    
    @Test
    void testSequentialIndices() {
        Dictionary dict = new Dictionary();
        
        // Indices should be sequential
        for (int i = 0; i < 100; i++) {
            assertEquals(i, dict.getIndex("item" + i));
        }
        
        // Existing items keep their indices
        for (int i = 0; i < 100; i++) {
            assertEquals(i, dict.getIndex("item" + i));
        }
        
        assertEquals(100, dict.size());
    }
    
    @Test
    void testLargeScale() {
        Dictionary dict = new Dictionary();
        final int numItems = 10000;
        
        // Add many items
        for (int i = 0; i < numItems; i++) {
            assertEquals(i, dict.getIndex("item" + i));
        }
        
        assertEquals(numItems, dict.size());
        
        // Verify random access
        assertEquals("item500", dict.getValue(500));
        assertEquals("item9999", dict.getValue(9999));
        
        // Verify contains
        assertTrue(dict.containsValue("item0"));
        assertTrue(dict.containsValue("item9999"));
        assertFalse(dict.containsValue("item10000"));
    }
}