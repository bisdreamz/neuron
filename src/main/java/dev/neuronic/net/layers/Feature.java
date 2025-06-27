package dev.neuronic.net.layers;

/**
 * Configuration for individual features in mixed feature input layers.
 * 
 * Each feature type is optimized for different data characteristics:
 * - High-cardinality features (bundle IDs, user IDs) use embeddings
 * - Low-cardinality features (device type, connection type) use one-hot encoding
 * - Numerical features (age, price) pass through unchanged
 */
public class Feature {
    
    public enum Type {
        EMBEDDING,           // High-cardinality → dense embeddings
        ONEHOT,              // Low-cardinality → one-hot vectors  
        PASSTHROUGH,         // Numerical → direct pass-through (no scaling)
        AUTO_NORMALIZE,      // Numerical → auto z-score normalize (mean=0, std=1)
        SCALE_BOUNDED        // Numerical → min-max scale with user-specified bounds
    }
    
    private final Type type;
    private final int maxUniqueValues;     // For EMBEDDING and ONEHOT types
    private final int embeddingDimension;  // For EMBEDDING type only
    private final float minBound;          // For SCALE_BOUNDED: user-specified minimum
    private final float maxBound;          // For SCALE_BOUNDED: user-specified maximum
    private final String name;             // Optional feature name for meaningful identification
    
    private Feature(Type type, int maxUniqueValues, int embeddingDimension) {
        this(type, maxUniqueValues, embeddingDimension, 0.0f, 0.0f, null);
    }
    
    private Feature(Type type, int maxUniqueValues, int embeddingDimension, float minBound, float maxBound) {
        this(type, maxUniqueValues, embeddingDimension, minBound, maxBound, null);
    }
    
    private Feature(Type type, int maxUniqueValues, int embeddingDimension, float minBound, float maxBound, String name) {
        this.type = type;
        this.maxUniqueValues = maxUniqueValues;
        this.embeddingDimension = embeddingDimension;
        this.minBound = minBound;
        this.maxBound = maxBound;
        this.name = name;
    }
    
    /**
     * High-cardinality feature encoded as dense embeddings.
     * 
     * <p><b>When to use:</b> Features with many unique values where you want to learn 
     * relationships between similar values. Examples: bundle IDs, publisher IDs, 
     * user IDs, product IDs.
     * 
     * <p><b>How it works:</b> Each unique value gets mapped to a dense vector that 
     * the model learns to optimize. Similar values will have similar vectors.
     * 
     * <p><b>Memory usage:</b> maxUniqueValues × embeddingDimension × 4 bytes
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // Bundle IDs: 100,000 possible bundles, each represented as 64-dimensional vector
     * Feature.embedding(100000, 64)
     * 
     * // Input: bundle_id = 12345 → outputs 64 floating-point values
     * }</pre>
     * 
     * @param maxUniqueValues maximum number of different values this feature can have
     *                       (e.g., 100000 for bundle IDs with values 0-99999)
     * @param embeddingDimension size of the dense vector representation 
     *                          (typically 32-128, higher for more complex relationships)
     * @return feature configuration for embedding encoding
     */
    public static Feature embedding(int maxUniqueValues, int embeddingDimension) {
        if (maxUniqueValues <= 0)
            throw new IllegalArgumentException("maxUniqueValues must be positive: " + maxUniqueValues);
        if (embeddingDimension <= 0)
            throw new IllegalArgumentException("embeddingDimension must be positive: " + embeddingDimension);
            
        return new Feature(Type.EMBEDDING, maxUniqueValues, embeddingDimension);
    }
    
    /**
     * High-cardinality feature encoded as dense embeddings with a meaningful name.
     * 
     * <p>Same as {@link #embedding(int, int)} but with an explicit feature name
     * for use with Map-based inputs in SimpleNet wrappers.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * Feature.embedding(100000, 64, "bundle_id")
     * // Now can use: model.train(Map.of("bundle_id", 12345), target)
     * }</pre>
     * 
     * @param maxUniqueValues maximum number of different values this feature can have
     * @param embeddingDimension size of the dense vector representation
     * @param name meaningful name for this feature (e.g., "bundle_id", "user_id")
     * @return feature configuration for embedding encoding with name
     */
    public static Feature embedding(int maxUniqueValues, int embeddingDimension, String name) {
        if (maxUniqueValues <= 0)
            throw new IllegalArgumentException("maxUniqueValues must be positive: " + maxUniqueValues);
        if (embeddingDimension <= 0)
            throw new IllegalArgumentException("embeddingDimension must be positive: " + embeddingDimension);
        if (name == null || name.trim().isEmpty())
            throw new IllegalArgumentException("Feature name cannot be null or empty");
            
        return new Feature(Type.EMBEDDING, maxUniqueValues, embeddingDimension, 0.0f, 0.0f, name.trim());
    }
    
    /**
     * Low-cardinality feature encoded as one-hot vectors.
     * 
     * <p><b>When to use:</b> Features with few unique values where each value is 
     * independent. Examples: connection type (wifi/4g/3g), device type (phone/tablet), 
     * day of week, hour of day.
     * 
     * <p><b>How it works:</b> Creates a vector with numberOfCategories elements, 
     * all zeros except one 1.0 at the position corresponding to the input value.
     * 
     * <p><b>Memory usage:</b> No learned parameters, just temporary computation buffers
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // Connection type: 4 possibilities (wifi=0, 4g=1, 3g=2, other=3)  
     * Feature.oneHot(4)
     * 
     * // Input: connection_type = 1 → outputs [0.0, 1.0, 0.0, 0.0]
     * }</pre>
     * 
     * @param numberOfCategories how many different values this feature can have
     *                          (input values must be in range 0 to numberOfCategories-1)
     * @return feature configuration for one-hot encoding
     */
    public static Feature oneHot(int numberOfCategories) {
        if (numberOfCategories <= 0)
            throw new IllegalArgumentException("numberOfCategories must be positive: " + numberOfCategories);
            
        return new Feature(Type.ONEHOT, numberOfCategories, 0);
    }
    
    /**
     * Low-cardinality feature encoded as one-hot vectors with a meaningful name.
     * 
     * <p>Same as {@link #oneHot(int)} but with an explicit feature name
     * for use with Map-based inputs in SimpleNet wrappers.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * Feature.oneHot(4, "connection_type")
     * // Now can use: model.train(Map.of("connection_type", 1), target)
     * }</pre>
     * 
     * @param numberOfCategories how many different values this feature can have
     * @param name meaningful name for this feature (e.g., "device_type", "day_of_week")
     * @return feature configuration for one-hot encoding with name
     */
    public static Feature oneHot(int numberOfCategories, String name) {
        if (numberOfCategories <= 0)
            throw new IllegalArgumentException("numberOfCategories must be positive: " + numberOfCategories);
        if (name == null || name.trim().isEmpty())
            throw new IllegalArgumentException("Feature name cannot be null or empty");
            
        return new Feature(Type.ONEHOT, numberOfCategories, 0, 0.0f, 0.0f, name.trim());
    }
    
    /**
     * Numerical feature passed through unchanged.
     * 
     * <p><b>When to use:</b> Continuous numerical values that are already properly scaled
     * or when you want manual control over scaling. Use this if your values are already
     * in a reasonable range (e.g., probabilities, percentages, normalized values).
     * 
     * <p><b>How it works:</b> Input value is copied directly to output without 
     * any transformation.
     * 
     * <p><b>Memory usage:</b> No learned parameters or buffers needed
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // CTR that's already a probability [0,1]
     * Feature.passthrough()
     * 
     * // Input: ctr = 0.023 → outputs 0.023
     * }</pre>
     * 
     * @return feature configuration for pass-through (no scaling)
     */
    public static Feature passthrough() {
        return new Feature(Type.PASSTHROUGH, 0, 0);
    }
    
    /**
     * Numerical feature passed through unchanged with a meaningful name.
     * 
     * <p>Same as {@link #passthrough()} but with an explicit feature name
     * for use with Map-based inputs in SimpleNet wrappers.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * Feature.passthrough("ctr")
     * // Now can use: model.train(Map.of("ctr", 0.023f), target)
     * }</pre>
     * 
     * @param name meaningful name for this feature (e.g., "price", "ctr", "age")
     * @return feature configuration for pass-through with name
     */
    public static Feature passthrough(String name) {
        if (name == null || name.trim().isEmpty())
            throw new IllegalArgumentException("Feature name cannot be null or empty");
            
        return new Feature(Type.PASSTHROUGH, 0, 0, 0.0f, 0.0f, name.trim());
    }
    
    /**
     * Numerical feature with min-max scaling to [0,1] range using user-specified bounds.
     * 
     * <p><b>When to use:</b> When you know the reasonable bounds for your feature
     * and want stable, predictable scaling. Perfect for business-constrained values
     * like bid floors, ages, percentages, etc.
     * 
     * <p><b>How it works:</b> Uses fixed bounds you specify. Scales values using:
     * (value - minBound) / (maxBound - minBound). Values outside bounds are
     * clamped to [0,1] range.
     * 
     * <p><b>Memory usage:</b> No runtime statistics needed - just stores bounds
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // Bid floor: business logic says $0.01 to $100.00 is reasonable range
     * Feature.autoScale(0.01f, 100.0f)
     * 
     * // Always produces predictable scaling:
     * // Input: bid_floor = $2.50 → outputs 0.0249 (always same result)
     * // Input: bid_floor = $150.00 → outputs 1.0 (clamped to max)
     * }</pre>
     * 
     * <p><b>Benefits:</b>
     * <ul>
     *   <li>Stable scaling - same input always produces same output</li>
     *   <li>Handles out-of-range values gracefully (clamps to [0,1])</li>
     *   <li>No "cold start" problem - works correctly from first prediction</li>
     *   <li>Incorporates business knowledge about reasonable value ranges</li>
     *   <li>Better for production systems with evolving data</li>
     * </ul>
     * 
     * @param minBound minimum expected value (will map to 0.0)
     * @param maxBound maximum expected value (will map to 1.0)
     * @return feature configuration for bounded min-max scaling
     * @throws IllegalArgumentException if minBound >= maxBound
     */
    public static Feature autoScale(float minBound, float maxBound) {
        if (minBound >= maxBound) {
            throw new IllegalArgumentException(String.format(
                "minBound (%.3f) must be less than maxBound (%.3f)", minBound, maxBound));
        }
        return new Feature(Type.SCALE_BOUNDED, 0, 0, minBound, maxBound);
    }
    
    /**
     * Numerical feature with min-max scaling using user-specified bounds and a meaningful name.
     * 
     * <p>Same as {@link #autoScale(float, float)} but with an explicit feature name
     * for use with Map-based inputs in SimpleNet wrappers.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * Feature.autoScale(0.01f, 100.0f, "bid_floor")
     * // Now can use: model.train(Map.of("bid_floor", 2.50f), target)
     * }</pre>
     * 
     * @param minBound minimum expected value (will map to 0.0)
     * @param maxBound maximum expected value (will map to 1.0)
     * @param name meaningful name for this feature (e.g., "bid_floor", "confidence_score")
     * @return feature configuration for bounded min-max scaling with name
     * @throws IllegalArgumentException if minBound >= maxBound or name is invalid
     */
    public static Feature autoScale(float minBound, float maxBound, String name) {
        if (minBound >= maxBound) {
            throw new IllegalArgumentException(String.format(
                "minBound (%.3f) must be less than maxBound (%.3f)", minBound, maxBound));
        }
        if (name == null || name.trim().isEmpty())
            throw new IllegalArgumentException("Feature name cannot be null or empty");
            
        return new Feature(Type.SCALE_BOUNDED, 0, 0, minBound, maxBound, name.trim());
    }
    
    
    /**
     * Numerical feature with automatic z-score normalization (mean=0, std=1).
     * 
     * <p><b>When to use:</b> Continuous numerical values that follow roughly normal
     * distributions. Good for features where you want to preserve the relative
     * distances between values while centering around zero.
     * 
     * <p><b>How it works:</b> During training, tracks running mean and standard deviation.
     * Normalizes all values using: (value - mean) / std. Handles edge cases where
     * std=0 by outputting 0.0.
     * 
     * <p><b>Memory usage:</b> Stores running mean/std statistics per feature
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // User age that typically ranges 18-65 with mean ~35
     * Feature.autoNormalize()
     * 
     * // After seeing ages with mean=35, std=12:
     * // Input: age = 47 → outputs 1.0 (one std dev above mean)
     * // Input: age = 23 → outputs -1.0 (one std dev below mean)
     * }</pre>
     * 
     * <p><b>Benefits:</b>
     * <ul>
     *   <li>Centers data around zero (good for neural networks)</li>
     *   <li>Preserves relative distances between values</li>
     *   <li>Works well with normally distributed data</li>
     *   <li>Standard approach in many ML applications</li>
     * </ul>
     * 
     * @return feature configuration for automatic z-score normalization
     */
    public static Feature autoNormalize() {
        return new Feature(Type.AUTO_NORMALIZE, 0, 0);
    }
    
    /**
     * Numerical feature with automatic z-score normalization and a meaningful name.
     * 
     * <p>Same as {@link #autoNormalize()} but with an explicit feature name
     * for use with Map-based inputs in SimpleNet wrappers.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * Feature.autoNormalize("user_age")
     * // Now can use: model.train(Map.of("user_age", 35.0f), target)
     * }</pre>
     * 
     * @param name meaningful name for this feature (e.g., "user_age", "time_spent")
     * @return feature configuration for automatic z-score normalization with name
     */
    public static Feature autoNormalize(String name) {
        if (name == null || name.trim().isEmpty())
            throw new IllegalArgumentException("Feature name cannot be null or empty");
            
        return new Feature(Type.AUTO_NORMALIZE, 0, 0, 0.0f, 0.0f, name.trim());
    }
    
    
    // Package-private getters for layer implementation
    public Type getType() { return type; }
    int getMaxUniqueValues() { return maxUniqueValues; }
    int getEmbeddingDimension() { return embeddingDimension; }
    float getMinBound() { return minBound; }
    float getMaxBound() { return maxBound; }
    public String getName() { return name; }
    
    /**
     * Calculate the output dimension for this feature.
     * - Embedding: embeddingDimension  
     * - OneHot: numberOfCategories
     * - Passthrough: 1
     * - AutoScale: 1
     * - AutoNormalize: 1
     */
    int getOutputDimension() {
        return switch (type) {
            case EMBEDDING -> embeddingDimension;
            case ONEHOT -> maxUniqueValues; // numberOfCategories stored in maxUniqueValues
            case PASSTHROUGH, AUTO_NORMALIZE, SCALE_BOUNDED -> 1;
        };
    }
    
    @Override
    public String toString() {
        String base = switch (type) {
            case EMBEDDING -> String.format("Feature.embedding(maxUniqueValues=%d, embeddingDimension=%d)", 
                                           maxUniqueValues, embeddingDimension);
            case ONEHOT -> String.format("Feature.oneHot(numberOfCategories=%d)", maxUniqueValues);
            case PASSTHROUGH -> "Feature.passthrough()";
            case AUTO_NORMALIZE -> "Feature.autoNormalize()";
            case SCALE_BOUNDED -> String.format("Feature.autoScale(%.3f, %.3f)", minBound, maxBound);
        };
        
        return name != null ? base + " [name=" + name + "]" : base;
    }
}