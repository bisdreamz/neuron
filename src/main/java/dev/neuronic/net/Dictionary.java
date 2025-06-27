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
    private final boolean allowNewValues;

    /**
     * Create a dictionary that allows new values to be added automatically.
     */
    public Dictionary() {
        this(true);
    }

    /**
     * Create a dictionary with configurable behavior for unknown values.
     *
     * @param allowNewValues if true, unknown values get new indices; if false, returns -1
     */
    public Dictionary(boolean allowNewValues) {
        this.allowNewValues = allowNewValues;
    }

    /**
     * Get the index for a value, creating a new index if value is unknown and allowed.
     *
     * @param value the value to look up
     * @return index for the value, or -1 if unknown and new values not allowed
     */
    public int getIndex(Object value) {
        if (value == null)
            throw new IllegalArgumentException("Dictionary values cannot be null");

        Integer existing = valueToIndex.get(value);
        if (existing != null)
            return existing;

        if (!allowNewValues)
            return -1; // Unknown value, not allowed to create new

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
     * Create a frozen copy of this dictionary that doesn't allow new values.
     * Useful for inference when you want to reject unknown values.
     */
    public Dictionary freeze() {
        Dictionary frozen = new Dictionary(false);

        frozen.valueToIndex.putAll(this.valueToIndex);
        frozen.indexToValue.putAll(this.indexToValue);
        frozen.nextIndex.set(this.nextIndex.get());

        return frozen;
    }

    /**
     * Serialize the dictionary to a stream.
     */
    public void writeTo(DataOutputStream out) throws IOException {
        out.writeInt(valueToIndex.size());
        out.writeBoolean(allowNewValues);
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
        boolean allowNewValues = in.readBoolean();
        int nextIndex = in.readInt();

        Dictionary dict = new Dictionary(allowNewValues);
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
        int size = 12; // size + allowNewValues + nextIndex
        for (Object key : valueToIndex.keySet()) {
            size += 2 + key.toString().getBytes().length; // UTF string
            size += 4; // integer value
        }
        return size;
    }

    @Override
    public String toString() {
        return String.format("Dictionary[size=%d, allowNewValues=%s, nextIndex=%d]",
                size(), allowNewValues, getNextIndex());
    }
}