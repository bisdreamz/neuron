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
        SCALE_BOUNDED,       // Numerical → min-max scale with user-specified bounds
        HASHED_EMBEDDING     // High-cardinality → hash-based embeddings (no vocabulary limit)
    }
    
    private final Type type;
    private final int maxUniqueValues;     // For EMBEDDING and ONEHOT types
    private final int embeddingDimension;  // For EMBEDDING type only
    private final float minBound;          // For SCALE_BOUNDED: user-specified minimum
    private final float maxBound;          // For SCALE_BOUNDED: user-specified maximum
    private final String name;             // Optional feature name for meaningful identification
    private final boolean useLRU;          // Whether to use LRU eviction for dictionary
    
    private Feature(Type type, int maxUniqueValues, int embeddingDimension) {
        this(type, maxUniqueValues, embeddingDimension, 0.0f, 0.0f, null, false);
    }
    
    private Feature(Type type, int maxUniqueValues, int embeddingDimension, float minBound, float maxBound) {
        this(type, maxUniqueValues, embeddingDimension, minBound, maxBound, null, false);
    }
    
    private Feature(Type type, int maxUniqueValues, int embeddingDimension, float minBound, float maxBound, String name) {
        this(type, maxUniqueValues, embeddingDimension, minBound, maxBound, name, false);
    }
    
    private Feature(Type type, int maxUniqueValues, int embeddingDimension, float minBound, float maxBound, String name, boolean useLRU) {
        this.type = type;
        this.maxUniqueValues = maxUniqueValues;
        this.embeddingDimension = embeddingDimension;
        this.minBound = minBound;
        this.maxBound = maxBound;
        this.name = name;
        this.useLRU = useLRU;
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
     * High-cardinality feature with LRU eviction for online learning.
     * 
     * <p><b>When to use LRU embeddings:</b>
     * <ul>
     *   <li>Online learning with evolving vocabulary</li>
     *   <li>Memory-constrained environments</li>
     *   <li>Features where recent values matter more than old ones</li>
     *   <li>Long-tail distributions where rare values can be forgotten</li>
     * </ul>
     * 
     * <p>When the dictionary reaches maxUniqueValues, the least recently used
     * entries are evicted to make room for new ones. This prevents unbounded
     * memory growth while adapting to changing data distributions.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // User IDs in online learning - keep 100k most recent users
     * Feature.embeddingLRU(100000, 64, "user_id")
     * 
     * // Product SKUs with seasonal changes
     * Feature.embeddingLRU(50000, 32, "product_sku")
     * }</pre>
     * 
     * @param maxUniqueValues maximum entries before LRU eviction starts
     * @param embeddingDimension size of the dense vector representation
     * @param name meaningful name for this feature
     * @return feature configuration for LRU embedding encoding
     */
    public static Feature embeddingLRU(int maxUniqueValues, int embeddingDimension, String name) {
        if (maxUniqueValues <= 0)
            throw new IllegalArgumentException("maxUniqueValues must be positive: " + maxUniqueValues);
        if (embeddingDimension <= 0)
            throw new IllegalArgumentException("embeddingDimension must be positive: " + embeddingDimension);
        if (name == null || name.trim().isEmpty())
            throw new IllegalArgumentException("Feature name cannot be null or empty");
            
        return new Feature(Type.EMBEDDING, maxUniqueValues, embeddingDimension, 0.0f, 0.0f, name.trim(), true);
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
     * Low-cardinality feature with LRU eviction for evolving categories.
     * 
     * <p><b>When to use LRU one-hot:</b>
     * <ul>
     *   <li>Categories that change over time (e.g., trending hashtags)</li>
     *   <li>Features with occasional new values in production</li>
     *   <li>When you want to cap memory usage for categorical features</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // Device OS versions - keep 20 most recent
     * Feature.oneHotLRU(20, "os_version")
     * 
     * // Error codes that evolve with new releases
     * Feature.oneHotLRU(100, "error_code")
     * }</pre>
     * 
     * @param numberOfCategories maximum categories before LRU eviction
     * @param name meaningful name for this feature
     * @return feature configuration for LRU one-hot encoding
     */
    public static Feature oneHotLRU(int numberOfCategories, String name) {
        if (numberOfCategories <= 0)
            throw new IllegalArgumentException("numberOfCategories must be positive: " + numberOfCategories);
        if (name == null || name.trim().isEmpty())
            throw new IllegalArgumentException("Feature name cannot be null or empty");
            
        return new Feature(Type.ONEHOT, numberOfCategories, 0, 0.0f, 0.0f, name.trim(), true);
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
    
    /**
     * Creates a hashed embedding feature for high-cardinality categorical data.
     * 
     * <p><b>When to use over regular embeddings:</b>
     * <table>
     *   <tr><th>Use hashedEmbedding when:</th><th>Use regular embedding when:</th></tr>
     *   <tr><td>Vocabulary size unknown</td><td>Fixed, known vocabulary</td></tr>
     *   <tr><td>Millions of possible values</td><td>Less than 10k values</td></tr>
     *   <tr><td>Online learning scenario</td><td>Batch training with preprocessing</td></tr>
     *   <tr><td>Memory constraints</td><td>Need perfect value distinction</td></tr>
     * </table>
     * 
     * <p><b>How it works:</b> Uses multiple hash functions to map strings to embedding
     * indices, then averages the embeddings. This provides collision resistance without
     * requiring a vocabulary dictionary.
     * 
     * <p><b>Recommended configurations:</b>
     * <ul>
     *   <li>Low cardinality (100-1k): {@code hashedEmbedding(1_000, 8, name)}</li>
     *   <li>Medium cardinality (1k-10k): {@code hashedEmbedding(10_000, 16, name)}</li>
     *   <li>High cardinality (10k-100k): {@code hashedEmbedding(50_000, 32, name)}</li>
     *   <li>Very high (100k+): {@code hashedEmbedding(100_000, 64, name)}</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // For domains with potentially millions of unique values
     * Feature.hashedEmbedding(10_000, 16, "domain")
     * 
     * // For app bundles with unknown vocabulary
     * Feature.hashedEmbedding(50_000, 32, "app_bundle")
     * }</pre>
     * 
     * @param hashBuckets number of hash buckets (100 to 1M)
     * @param embeddingDim embedding dimension (4 to 256)
     * @param name feature name for debugging
     * @return hashed embedding feature
     * @throws IllegalArgumentException if parameters are out of valid ranges
     */
    public static Feature hashedEmbedding(int hashBuckets, int embeddingDim, String name) {
        // Validation
        if (hashBuckets < 100)
            throw new IllegalArgumentException(
                "Hash buckets too small: " + hashBuckets + ". Minimum 100 to avoid excessive collisions.");
        if (hashBuckets > 1_000_000)
            throw new IllegalArgumentException(
                "Hash buckets too large: " + hashBuckets + ". Maximum 1M to avoid memory issues.");
        if (embeddingDim < 4)
            throw new IllegalArgumentException(
                "Embedding dimension too small: " + embeddingDim + ". Minimum 4 for meaningful representations.");
        if (embeddingDim > 256)
            throw new IllegalArgumentException(
                "Embedding dimension too large: " + embeddingDim + ". Maximum 256 to avoid overfitting.");
        
        // Check parameter efficiency
        long totalParams = (long) hashBuckets * embeddingDim;
        if (totalParams > 50_000_000)
            throw new IllegalArgumentException(String.format(
                "Total parameters (%d buckets × %d dims = %,d) exceeds 50M limit. " +
                "Reduce buckets or embedding dimension.", 
                hashBuckets, embeddingDim, totalParams));
        
        // Warn about dimension/bucket ratio
        if (embeddingDim > hashBuckets / 100)
            throw new IllegalArgumentException(String.format(
                "Embedding dimension (%d) too large relative to buckets (%d). " +
                "Dimension should be < buckets/100 for efficiency.", 
                embeddingDim, hashBuckets));
        
        if (name == null || name.trim().isEmpty())
            throw new IllegalArgumentException("Feature name cannot be null or empty");
            
        return new Feature(Type.HASHED_EMBEDDING, hashBuckets, embeddingDim, 0.0f, 0.0f, name.trim());
    }
    
    
    // Getters for layer implementation and validation
    public Type getType() { return type; }
    public int getMaxUniqueValues() { return maxUniqueValues; }
    int getEmbeddingDimension() { return embeddingDimension; }
    float getMinBound() { return minBound; }
    float getMaxBound() { return maxBound; }
    public String getName() { return name; }
    public boolean isLRU() { return useLRU; }
    
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
            case EMBEDDING, HASHED_EMBEDDING -> embeddingDimension;
            case ONEHOT -> maxUniqueValues; // numberOfCategories stored in maxUniqueValues
            case PASSTHROUGH, AUTO_NORMALIZE, SCALE_BOUNDED -> 1;
        };
    }
    
    @Override
    public String toString() {
        String base = switch (type) {
            case EMBEDDING -> String.format("Feature.embedding%s(maxUniqueValues=%d, embeddingDimension=%d)", 
                                           useLRU ? "LRU" : "", maxUniqueValues, embeddingDimension);
            case HASHED_EMBEDDING -> String.format("Feature.hashedEmbedding(hashBuckets=%d, embeddingDimension=%d)", 
                                           maxUniqueValues, embeddingDimension);
            case ONEHOT -> String.format("Feature.oneHot%s(numberOfCategories=%d)", 
                                        useLRU ? "LRU" : "", maxUniqueValues);
            case PASSTHROUGH -> "Feature.passthrough()";
            case AUTO_NORMALIZE -> "Feature.autoNormalize()";
            case SCALE_BOUNDED -> String.format("Feature.autoScale(%.3f, %.3f)", minBound, maxBound);
        };
        
        return name != null ? base + " [name=" + name + "]" : base;
    }
}