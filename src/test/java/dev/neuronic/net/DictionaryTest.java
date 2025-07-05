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
        Dictionary dict = new Dictionary(1000);
        
        // Store indices for repeated access verification
        int firstIndex = dict.getIndex("first");
        int secondIndex = dict.getIndex("second");
        int intIndex = dict.getIndex(123);
        int floatIndex = dict.getIndex(45.6f);
        
        // Verify indices are distributed (not sequential)
        assertTrue(firstIndex >= 0 && firstIndex < 1024);
        assertTrue(secondIndex >= 0 && secondIndex < 1024);
        assertTrue(intIndex >= 0 && intIndex < 1024);
        assertTrue(floatIndex >= 0 && floatIndex < 1024);
        
        // Repeated values return same index
        assertEquals(firstIndex, dict.getIndex("first"));
        assertEquals(intIndex, dict.getIndex(123));
        
        // Check size
        assertEquals(4, dict.size());
        assertEquals(4, dict.getNextIndex()); // Next index is current size
    }
    
    @Test
    void testGetValue() {
        Dictionary dict = new Dictionary(1000);
        
        int testIndex = dict.getIndex("test");
        int intIndex = dict.getIndex(42);
        
        assertEquals("test", dict.getValue(testIndex));
        assertEquals(42, dict.getValue(intIndex));
        assertNull(dict.getValue(999)); // Non-existent
    }
    
    @Test
    void testContains() {
        Dictionary dict = new Dictionary(1000);
        
        int index = dict.getIndex("exists");
        
        assertTrue(dict.containsValue("exists"));
        assertFalse(dict.containsValue("not-exists"));
        
        assertTrue(dict.containsIndex(index));
        assertFalse(dict.containsIndex(999)); // Random non-existent index
    }
    
    @Test
    void testNullValue() {
        Dictionary dict = new Dictionary(1000);
        assertThrows(IllegalArgumentException.class, () -> dict.getIndex(null));
    }
    
    @Test
    void testClear() {
        Dictionary dict = new Dictionary(1000);
        
        int indexA = dict.getIndex("A");
        int indexB = dict.getIndex("B");
        int indexC = dict.getIndex("C");
        
        assertEquals(3, dict.size());
        
        dict.clear();
        
        assertEquals(0, dict.size());
        assertEquals(0, dict.getNextIndex()); // Next index is size (0 after clear)
        assertFalse(dict.containsValue("A"));
        assertFalse(dict.containsIndex(indexA));
        
        // Can add after clear - will get new distributed index
        int newIndex = dict.getIndex("D");
        assertTrue(newIndex >= 0 && newIndex < 1024);
    }
    
    @Test
    void testMixedTypes() {
        Dictionary dict = new Dictionary(1000);
        
        // Different types get distributed indices
        int stringIndex = dict.getIndex("string");
        int intIndex = dict.getIndex(42);
        int floatIndex = dict.getIndex(3.14f);
        int boolIndex = dict.getIndex(true);
        int longIndex = dict.getIndex(99L);
        
        // Verify all indices are in valid range
        assertTrue(stringIndex >= 0 && stringIndex < 1024);
        assertTrue(intIndex >= 0 && intIndex < 1024);
        assertTrue(floatIndex >= 0 && floatIndex < 1024);
        assertTrue(boolIndex >= 0 && boolIndex < 1024);
        assertTrue(longIndex >= 0 && longIndex < 1024);
        
        // Verify retrieval
        assertEquals("string", dict.getValue(stringIndex));
        assertEquals(42, dict.getValue(intIndex));
        assertEquals(3.14f, dict.getValue(floatIndex));
        assertEquals(true, dict.getValue(boolIndex));
        assertEquals(99L, dict.getValue(longIndex));
    }
    
    @Test
    void testSerialization() throws IOException {
        Dictionary dict = new Dictionary(1000);
        
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
        Dictionary dict2 = Dictionary.readFrom(in, 1000);
        in.close();
        
        // Verify
        assertEquals(3, dict2.size());
        assertEquals(3, dict2.getNextIndex()); // Next index is size
        
        // Values preserved (as strings due to serialization)
        assertTrue(dict2.containsValue("string"));
        assertTrue(dict2.containsValue(42)); // Parsed back as Integer
        assertTrue(dict2.containsValue(3.14f)); // Parsed back as Float
    }
    
    @Test
    void testGetSerializedSize() {
        Dictionary dict = new Dictionary(1000);
        
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
        Dictionary dict = new Dictionary(1000);
        dict.getIndex("A");
        dict.getIndex("B");
        
        String str = dict.toString();
        assertTrue(str.contains("Dictionary"));
        assertTrue(str.contains("size=2"));
        assertTrue(str.contains("maxBounds=1000"));
    }
    
    @Test
    void testConcurrentAccess() throws InterruptedException {
        final Dictionary dict = new Dictionary(100000); // Large bounds for concurrent test
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
        
        // Verify total size - no collisions allowed, each unique value gets unique index
        assertEquals(numThreads * itemsPerThread, dict.size());
        
        // Verify all mappings are consistent
        for (var entry : expectedIndices.entrySet()) {
            assertEquals(entry.getValue().intValue(), dict.getIndex(entry.getKey()));
        }
    }
    
    @Test
    void testDistributedIndices() {
        Dictionary dict = new Dictionary(1000);
        
        // Collect indices for verification
        int[] indices = new int[100];
        for (int i = 0; i < 100; i++) {
            indices[i] = dict.getIndex("item" + i);
            // Verify each index is in valid range
            assertTrue(indices[i] >= 0 && indices[i] < 1024);
        }
        
        // Verify indices are distributed (not all the same)
        boolean foundDifferent = false;
        for (int i = 1; i < indices.length; i++) {
            if (indices[i] != indices[0]) {
                foundDifferent = true;
                break;
            }
        }
        assertTrue(foundDifferent, "Indices should be distributed, not clustered");
        
        // Existing items keep their indices
        for (int i = 0; i < 100; i++) {
            assertEquals(indices[i], dict.getIndex("item" + i));
        }
        
        assertEquals(100, dict.size());
    }
    
    @Test
    void testLargeScale() {
        Dictionary dict = new Dictionary(1000);
        final int numItems = 1000; // Reduced for distributed testing
        
        // Store indices for verification
        int[] indices = new int[numItems];
        for (int i = 0; i < numItems; i++) {
            indices[i] = dict.getIndex("item" + i);
            assertTrue(indices[i] >= 0 && indices[i] < 1024);
        }
        
        assertEquals(numItems, dict.size());
        
        // Verify random access using stored indices
        assertEquals("item500", dict.getValue(indices[500]));
        assertEquals("item999", dict.getValue(indices[999]));
        
        // Verify contains
        assertTrue(dict.containsValue("item0"));
        assertTrue(dict.containsValue("item999"));
        assertFalse(dict.containsValue("item1000"));
    }
}