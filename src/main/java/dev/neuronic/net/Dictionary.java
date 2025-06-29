package dev.neuronic.net;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Map;

/**
 * Thread-safe dictionary for mapping arbitrary values to integer indices.
 * Handles automatic index assignment and bi-directional lookup.
 */
public class Dictionary {

    private final ConcurrentHashMap<Object, Integer> valueToIndex = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, Object> indexToValue = new ConcurrentHashMap<>();
    private final AtomicInteger nextIndex = new AtomicInteger(0);

    /**
     * Create a dictionary that allows new values to be added automatically.
     */
    public Dictionary() {
    }

    /**
     * Get the index for a value, creating a new index if value is unknown.
     *
     * @param value the value to look up
     * @return index for the value
     */
    public int getIndex(Object value) {
        if (value == null)
            throw new IllegalArgumentException("Dictionary values cannot be null");

        Integer existing = valueToIndex.get(value);
        if (existing != null)
            return existing;

        int newIndex = nextIndex.getAndIncrement();
        Integer previousIndex = valueToIndex.putIfAbsent(value, newIndex);

        if (previousIndex != null)
            return previousIndex;

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
     * Get the next index that would be assigned to a new value.
     */
    public int getNextIndex() {
        return nextIndex.get();
    }

    /**
     * Clear all mappings from the dictionary.
     */
    public void clear() {
        valueToIndex.clear();
        indexToValue.clear();
        nextIndex.set(0);
    }


    /**
     * Serialize the dictionary to a stream.
     */
    public void writeTo(DataOutputStream out) throws IOException {
        out.writeInt(valueToIndex.size());
        out.writeInt(nextIndex.get());

        for (Map.Entry<Object, Integer> entry : valueToIndex.entrySet()) {
            // Write value as string representation for simplicity
            out.writeUTF(entry.getKey().toString());
            out.writeInt(entry.getValue());
        }
    }

    /**
     * Deserialize a dictionary from a stream.
     */
    public static Dictionary readFrom(DataInputStream in) throws IOException {
        int size = in.readInt();
        int nextIndex = in.readInt();

        Dictionary dict = new Dictionary();
        dict.nextIndex.set(nextIndex);

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
        int size = 8; // size + nextIndex
        for (Object key : valueToIndex.keySet()) {
            size += 2 + key.toString().getBytes().length; // UTF string
            size += 4; // integer value
        }
        return size;
    }

    @Override
    public String toString() {
        return String.format("Dictionary[size=%d, nextIndex=%d]",
                size(), getNextIndex());
    }
}