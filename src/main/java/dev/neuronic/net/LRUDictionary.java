package dev.neuronic.net;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Thread-safe dictionary with LRU (Least Recently Used) eviction policy.
 * 
 * <p>Maintains a fixed maximum size with automatic eviction of least recently used entries.
 * Uses sequential indexing with index reuse for evicted entries.
 * 
 * <p><b>Key features:</b>
 * <ul>
 *   <li>Fixed maximum size with automatic LRU eviction</li>
 *   <li>Thread-safe for concurrent access</li>
 *   <li>Maintains access order for LRU eviction</li>
 *   <li>Bi-directional lookup (value→index, index→value)</li>
 *   <li>Index reuse for evicted entries to stay within bounds</li>
 * </ul>
 * 
 * <p><b>When to use LRUDictionary:</b>
 * <ul>
 *   <li>Online learning with evolving vocabulary</li>
 *   <li>Limited memory scenarios</li>
 *   <li>Features with long-tail distributions</li>
 *   <li>When recent values are more important than old ones</li>
 * </ul>
 * 
 * <p><b>Trade-offs:</b>
 * <ul>
 *   <li>Slightly higher overhead than regular Dictionary due to access tracking</li>
 *   <li>Previously seen values may be forgotten if evicted</li>
 * </ul>
 */
public class LRUDictionary extends Dictionary {
    
    private final int maxSize;
    private final LinkedHashMap<Object, Integer> lruMap;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    
    /**
     * Create an LRU dictionary with specified maximum size and index bounds.
     * 
     * @param maxSize maximum number of entries before eviction begins
     * @param maxBounds maximum index value allowed (exclusive) 
     * @throws IllegalArgumentException if parameters are not positive
     */
    public LRUDictionary(int maxSize, int maxBounds) {
        super(maxBounds);
        if (maxSize <= 0)
            throw new IllegalArgumentException("Max size must be positive: " + maxSize);
            
        this.maxSize = maxSize;
        
        // Create LinkedHashMap with access order (true = access-order, false = insertion-order)
        this.lruMap = new LinkedHashMap<Object, Integer>(16, 0.75f, true);
    }
    
    /**
     * Get the index for a value, creating an index if value is unknown.
     * Updates access order for LRU tracking.
     * 
     * @param value the value to look up
     * @return index for the value
     */
    @Override
    public int getIndex(Object value) {
        if (value == null)
            throw new IllegalArgumentException("Dictionary values cannot be null");
        
        lock.readLock().lock();
        try {
            Integer existing = lruMap.get(value);
            if (existing != null)
                return existing;
        } finally {
            lock.readLock().unlock();
        }
        
        // Need to add new value with distributed indexing
        lock.writeLock().lock();
        try {
            // Double-check after acquiring write lock
            Integer existing = lruMap.get(value);
            if (existing != null)
                return existing;
                
            // Check if we need to evict
            if (lruMap.size() >= maxSize) {
                // Find the least recently used entry
                Map.Entry<Object, Integer> eldest = lruMap.entrySet().iterator().next();
                Object evictedValue = eldest.getKey();
                int evictedIndex = eldest.getValue();
                
                lruMap.remove(evictedValue);
                // Remove from parent Dictionary data structures
                super.valueToIndex.remove(evictedValue);
                super.indexToValue.remove(evictedIndex);
                
                // Reuse the evicted index for the new value
                lruMap.put(value, evictedIndex);
                // Update parent Dictionary data structures
                super.valueToIndex.put(value, evictedIndex);
                super.indexToValue.put(evictedIndex, value);
                return evictedIndex;
            } else {
                // Still have room, use sequential index
                int newIndex = lruMap.size();
                
                lruMap.put(value, newIndex);
                // Update parent Dictionary data structures
                super.valueToIndex.put(value, newIndex);
                super.indexToValue.put(newIndex, value);
                return newIndex;
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    
    /**
     * Get the value for an index.
     * Does NOT update access order (only getIndex does).
     * 
     * @param index the index to look up
     * @return the value, or null if index not found
     */
    @Override
    public Object getValue(int index) {
        return super.getValue(index);
    }
    
    /**
     * Check if the dictionary contains a value.
     * Updates access order for LRU tracking.
     */
    @Override
    public boolean containsValue(Object value) {
        lock.readLock().lock();
        try {
            return lruMap.containsKey(value);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Check if the dictionary contains an index.
     */
    @Override
    public boolean containsIndex(int index) {
        return super.containsIndex(index);
    }
    
    /**
     * Get the current size of the dictionary.
     */
    @Override
    public int size() {
        lock.readLock().lock();
        try {
            return lruMap.size();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get the maximum size of this LRU dictionary.
     * 
     * @return the maximum number of entries before eviction
     */
    public int getMaxSize() {
        return maxSize;
    }
    
    /**
     * Get the next index that would be assigned to a new value.
     * For LRU dictionary, this is the current size unless at capacity.
     */
    @Override
    public int getNextIndex() {
        lock.readLock().lock();
        try {
            return Math.min(lruMap.size(), maxSize);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Clear all mappings from the dictionary.
     */
    @Override
    public void clear() {
        lock.writeLock().lock();
        try {
            lruMap.clear();
            // Clear parent class data structures
            super.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Serialize the LRU dictionary to a stream.
     * Note: Does NOT preserve access order, only current mappings.
     */
    @Override
    public void writeTo(DataOutputStream out) throws IOException {
        lock.readLock().lock();
        try {
            out.writeInt(maxSize);
            out.writeInt(lruMap.size());
            
            for (Map.Entry<Object, Integer> entry : lruMap.entrySet()) {
                out.writeUTF(entry.getKey().toString());
                out.writeInt(entry.getValue());
            }
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Deserialize an LRU dictionary from a stream.
     * Note: Access order starts fresh after deserialization.
     */
    public static LRUDictionary readFrom(DataInputStream in, int maxBounds) throws IOException {
        int maxSize = in.readInt();
        int size = in.readInt();
        
        LRUDictionary dict = new LRUDictionary(maxSize, maxBounds);
        
        for (int i = 0; i < size; i++) {
            String valueStr = in.readUTF();
            int index = in.readInt();
            
            // Try to parse as different types (simple heuristic)
            Object value = parseValue(valueStr);
            dict.lock.writeLock().lock();
            try {
                dict.lruMap.put(value, index);
                dict.valueToIndex.put(value, index);
                dict.indexToValue.put(index, value);
            } finally {
                dict.lock.writeLock().unlock();
            }
        }
        
        return dict;
    }
    
    private static Object parseValue(String str) {
        try {
            return Integer.parseInt(str);
        } catch (NumberFormatException e) {
            try {
                return Float.parseFloat(str);
            } catch (NumberFormatException e2) {
                return str;
            }
        }
    }
    
    /**
     * Get the estimated serialized size of this dictionary.
     */
    @Override
    public int getSerializedSize() {
        lock.readLock().lock();
        try {
            int size = 8; // maxSize + size
            for (Object key : lruMap.keySet()) {
                size += 2 + key.toString().getBytes().length; // UTF string
                size += 4; // integer value
            }
            return size;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public String toString() {
        lock.readLock().lock();
        try {
            return String.format("LRUDictionary[size=%d/%d]",
                    lruMap.size(), maxSize);
        } finally {
            lock.readLock().unlock();
        }
    }
}