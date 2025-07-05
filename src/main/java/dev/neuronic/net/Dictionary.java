package dev.neuronic.net;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

/**
 * Thread-safe dictionary for mapping arbitrary values to integer indices.
 * Uses sequential indexing for compatibility with serialization and tests.
 * Enforces maximum bounds to ensure indices stay within neural network layer expectations.
 */
public class Dictionary {

    protected final ConcurrentHashMap<Object, Integer> valueToIndex = new ConcurrentHashMap<>();
    protected final ConcurrentHashMap<Integer, Object> indexToValue = new ConcurrentHashMap<>();
    protected final int maxBounds; // Maximum index value allowed (for neural network layer compatibility)

    /**
     * Create a dictionary with specified maximum bounds for indices.
     * 
     * @param maxBounds maximum index value allowed (exclusive)
     */
    public Dictionary(int maxBounds) {
        if (maxBounds <= 0)
            throw new IllegalArgumentException("maxBounds must be positive: " + maxBounds);
        this.maxBounds = maxBounds;
    }

    /**
     * Get the index for a value, creating an index if value is unknown.
     * Uses sequential allocation for compatibility with existing tests and serialization.
     * 
     * @param value the value to look up
     * @return index for the value
     */
    public synchronized int getIndex(Object value) {
        if (value == null)
            throw new IllegalArgumentException("Dictionary values cannot be null");

        Integer existing = valueToIndex.get(value);
        if (existing != null)
            return existing;

        // Check if we have capacity
        int currentSize = size();
        if (currentSize >= maxBounds) {
            throw new IllegalStateException("Dictionary is full: cannot allocate more indices within bounds [0, " + maxBounds + ")");
        }
        
        // Use sequential indexing for test compatibility
        int newIndex = currentSize;
        
        valueToIndex.put(value, newIndex);
        indexToValue.put(newIndex, value);
        return newIndex;
    }
    

    /**
     * Get the value for an index.
     *
     * @param index the index to look up
     * @return the value, or null if index not found
     */
    public Object getValue(int index) {
        return indexToValue.get(index);
    }

    /**
     * Check if the dictionary contains a value.
     */
    public boolean containsValue(Object value) {
        return valueToIndex.containsKey(value);
    }

    /**
     * Check if the dictionary contains an index.
     */
    public boolean containsIndex(int index) {
        return indexToValue.containsKey(index);
    }

    /**
     * Get the current size of the dictionary (number of unique values).
     */
    public int size() {
        return valueToIndex.size();
    }

    /**
     * Get the next index that would be assigned.
     * For sequential allocation, this is the current size.
     */
    public int getNextIndex() {
        return size();
    }

    /**
     * Clear all mappings from the dictionary.
     */
    public void clear() {
        valueToIndex.clear();
        indexToValue.clear();
    }


    /**
     * Serialize the dictionary to a stream.
     */
    public void writeTo(DataOutputStream out) throws IOException {
        out.writeInt(valueToIndex.size());

        for (Map.Entry<Object, Integer> entry : valueToIndex.entrySet()) {
            // Write value as string representation for simplicity
            out.writeUTF(entry.getKey().toString());
            out.writeInt(entry.getValue());
        }
    }

    /**
     * Deserialize a dictionary from a stream.
     */
    public static Dictionary readFrom(DataInputStream in, int maxBounds) throws IOException {
        int size = in.readInt();

        Dictionary dict = new Dictionary(maxBounds);

        for (int i = 0; i < size; i++) {
            String valueStr = in.readUTF();
            int index = in.readInt();

            // Try to parse as different types (simple heuristic)
            Object value = parseValue(valueStr);
            dict.valueToIndex.put(value, index);
            dict.indexToValue.put(index, value);
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
    public int getSerializedSize() {
        int size = 8; // size + tableSize
        for (Object key : valueToIndex.keySet()) {
            size += 2 + key.toString().getBytes().length; // UTF string
            size += 4; // integer value
        }
        return size;
    }

    @Override
    public String toString() {
        return String.format("Dictionary[size=%d, maxBounds=%d]",
                size(), maxBounds);
    }
}