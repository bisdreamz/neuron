package dev.neuronic.net.simple;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * Iterator for training data that allows streaming large datasets without loading
 * everything into memory.
 * 
 * <p><b>Quick Start - Stream from file:</b>
 * <pre>{@code
 * // Language model training from large text file
 * model.trainBulk(
 *     DataIterator.fromFile("huge_dataset.txt", line -> {
 *         String[] parts = line.split("\t");
 *         return DataBatch.single(
 *             Arrays.copyOf(parts, parts.length - 1),  // sequence
 *             parts[parts.length - 1]                   // next word
 *         );
 *     }),
 *     config
 * );
 * }</pre>
 * 
 * <p><b>Stream from database:</b>
 * <pre>{@code
 * model.trainBulk(
 *     DataIterator.fromSupplier(
 *         batchSize -> {
 *             // Fetch next batch from database
 *             ResultSet rs = stmt.executeQuery(
 *                 "SELECT sequence, target FROM data LIMIT " + batchSize + 
 *                 " OFFSET " + currentOffset
 *             );
 *             currentOffset += batchSize;
 *             
 *             List<float[]> inputs = new ArrayList<>();
 *             List<Integer> targets = new ArrayList<>();
 *             while (rs.next()) {
 *                 inputs.add(parseFeatures(rs.getString("sequence")));
 *                 targets.add(rs.getInt("target"));
 *             }
 *             return new DataBatch<>(inputs, targets);
 *         },
 *         () -> currentOffset < totalRows
 *     ),
 *     config
 * );
 * }</pre>
 * 
 * @param <I> input type (e.g., float[], String[], Map<String,Object>)
 * @param <T> target type (e.g., Integer, String, Float)
 */
public interface DataIterator<I, T> extends Iterator<DataBatch<I, T>>, AutoCloseable {
    
    /**
     * Get the next batch of data.
     * 
     * @param batchSize requested batch size (actual may be smaller for last batch)
     * @return next batch of data
     */
    DataBatch<I, T> nextBatch(int batchSize);
    
    /**
     * Check if more data is available.
     * 
     * @return true if more data available
     */
    @Override
    boolean hasNext();
    
    /**
     * Get next single element (for compatibility with Iterator).
     * 
     * @return next data batch (typically size 1)
     */
    @Override
    default DataBatch<I, T> next() {
        return nextBatch(1);
    }
    
    /**
     * Reset the iterator to start from beginning if supported.
     * 
     * @throws UnsupportedOperationException if reset not supported
     */
    default void reset() {
        throw new UnsupportedOperationException("Reset not supported by this iterator");
    }
    
    /**
     * Get estimated total number of elements if known.
     * 
     * @return total elements or -1 if unknown
     */
    default long estimatedSize() {
        return -1;
    }
    
    /**
     * Close any resources held by this iterator.
     */
    @Override
    default void close() {
        // Default: nothing to close
    }
    
    // ===============================
    // FACTORY METHODS
    // ===============================
    
    /**
     * Create an iterator from a supplier function.
     * 
     * <p><b>Example - Random data generation:</b>
     * <pre>{@code
     * DataIterator<float[], Integer> randomData = DataIterator.fromSupplier(
     *     batchSize -> {
     *         float[][] inputs = new float[batchSize][784];
     *         Integer[] labels = new Integer[batchSize];
     *         for (int i = 0; i < batchSize; i++) {
     *             // Generate random MNIST-like data
     *             inputs[i] = generateRandomPixels();
     *             labels[i] = random.nextInt(10);
     *         }
     *         return new DataBatch<>(inputs, labels);
     *     },
     *     () -> true  // Infinite generator
     * );
     * }</pre>
     * 
     * @param batchSupplier function that supplies next batch
     * @param hasNext supplier that returns true if more data available
     * @return new data iterator
     */
    static <I, T> DataIterator<I, T> fromSupplier(
            Function<Integer, DataBatch<I, T>> batchSupplier,
            Supplier<Boolean> hasNext) {
        return new DataIterator<I, T>() {
            @Override
            public DataBatch<I, T> nextBatch(int batchSize) {
                if (!hasNext())
                    throw new IllegalStateException("No more data available");
                return batchSupplier.apply(batchSize);
            }
            
            @Override
            public boolean hasNext() {
                return hasNext.get();
            }
        };
    }
    
    /**
     * Create an iterator from a Java Stream.
     * 
     * <p><b>Example - Process CSV lines:</b>
     * <pre>{@code
     * DataIterator<float[], String> csvData = DataIterator.fromStream(
     *     Files.lines(Paths.get("data.csv")),
     *     line -> {
     *         String[] parts = line.split(",");
     *         float[] features = parseFeatures(parts[0..n-1]);
     *         String label = parts[n-1];
     *         return DataBatch.single(features, label);
     *     }
     * );
     * }</pre>
     * 
     * @param stream source stream
     * @param mapper function to convert stream elements to data batches
     * @return new data iterator
     */
    static <S, I, T> DataIterator<I, T> fromStream(
            Stream<S> stream,
            Function<S, DataBatch<I, T>> mapper) {
        Iterator<DataBatch<I, T>> streamIterator = stream.map(mapper).iterator();
        
        return new DataIterator<I, T>() {
            @Override
            public DataBatch<I, T> nextBatch(int batchSize) {
                if (batchSize == 1 && streamIterator.hasNext()) {
                    return streamIterator.next();
                }
                
                // Collect multiple elements for larger batch
                List<I> inputs = new ArrayList<>(batchSize);
                List<T> targets = new ArrayList<>(batchSize);
                
                for (int i = 0; i < batchSize && streamIterator.hasNext(); i++) {
                    DataBatch<I, T> single = streamIterator.next();
                    inputs.add(single.getInputs()[0]);
                    targets.add(single.getTargets()[0]);
                }
                
                if (inputs.isEmpty())
                    throw new IllegalStateException("No more data available");
                
                @SuppressWarnings("unchecked")
                I[] inputArray = (I[]) inputs.toArray();
                @SuppressWarnings("unchecked")
                T[] targetArray = (T[]) targets.toArray();
                
                return new DataBatch<>(inputArray, targetArray);
            }
            
            @Override
            public boolean hasNext() {
                return streamIterator.hasNext();
            }
            
            @Override
            public void close() {
                stream.close();
            }
        };
    }
    
    /**
     * Create an iterator from a file using a line parser.
     * 
     * <p><b>Example - Language model training:</b>
     * <pre>{@code
     * DataIterator<String[], String> fileData = DataIterator.fromFile(
     *     "training_data.txt",
     *     line -> {
     *         String[] tokens = line.split(" ");
     *         String[] sequence = Arrays.copyOf(tokens, tokens.length - 1);
     *         String target = tokens[tokens.length - 1];
     *         return DataBatch.single(sequence, target);
     *     }
     * );
     * }</pre>
     * 
     * @param filePath path to file
     * @param lineParser function to parse each line into a batch
     * @return new data iterator
     */
    static <I, T> DataIterator<I, T> fromFile(
            String filePath,
            Function<String, DataBatch<I, T>> lineParser) {
        try {
            return fromStream(
                Files.lines(Paths.get(filePath)),
                lineParser
            );
        } catch (IOException e) {
            throw new RuntimeException("Failed to open file: " + filePath, e);
        }
    }
    
    /**
     * Create an iterator from existing lists (with memory-efficient chunking).
     * 
     * <p><b>Example - Chunk large lists:</b>
     * <pre>{@code
     * // Even with lists, iterate in chunks to be memory-friendly
     * DataIterator<float[], Integer> chunked = DataIterator.fromLists(
     *     largeInputList,
     *     largeLabelList,
     *     1000  // Process 1000 at a time
     * );
     * }</pre>
     * 
     * @param inputs list of inputs
     * @param targets list of targets
     * @param chunkSize size of chunks to iterate
     * @return new data iterator
     */
    static <I, T> DataIterator<I, T> fromLists(
            List<I> inputs,
            List<T> targets,
            int chunkSize) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException(
                "Inputs and targets must have same size: " + 
                inputs.size() + " vs " + targets.size());
        }
        
        return new DataIterator<I, T>() {
            private int position = 0;
            
            @Override
            public DataBatch<I, T> nextBatch(int batchSize) {
                int actualBatchSize = Math.min(batchSize, inputs.size() - position);
                if (actualBatchSize <= 0)
                    throw new IllegalStateException("No more data available");
                
                List<I> batchInputs = inputs.subList(position, position + actualBatchSize);
                List<T> batchTargets = targets.subList(position, position + actualBatchSize);
                
                position += actualBatchSize;
                
                @SuppressWarnings("unchecked")
                I[] inputArray = (I[]) batchInputs.toArray();
                @SuppressWarnings("unchecked")
                T[] targetArray = (T[]) batchTargets.toArray();
                
                return new DataBatch<>(inputArray, targetArray);
            }
            
            @Override
            public boolean hasNext() {
                return position < inputs.size();
            }
            
            @Override
            public void reset() {
                position = 0;
            }
            
            @Override
            public long estimatedSize() {
                return inputs.size();
            }
        };
    }
}