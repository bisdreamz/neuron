package dev.neuronic.net.layers;

import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.common.PooledFloatArray;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.AdamOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationRegistry;
import dev.neuronic.net.serialization.SerializationService;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;

/**
 * Mixed feature input layer for advertising and recommendation systems.
 * 
 * <p>Efficiently handles different types of features in a single layer:
 * <ul>
 *   <li><b>High-cardinality features:</b> Bundle IDs, publisher IDs → dense embeddings</li>
 *   <li><b>Low-cardinality features:</b> Connection type, device type → one-hot encoding</li>
 *   <li><b>Numerical features:</b> Age, price, duration → pass-through, auto-scale, or auto-normalize</li>
 * </ul>
 * 
 * <p><b>Key Benefits:</b>
 * <ul>
 *   <li><b>Memory efficient:</b> Only allocates embedding tables where needed</li>
 *   <li><b>Type safety:</b> Feature configuration array position = input data position</li>
 *   <li><b>Performance:</b> Vectorized operations with zero-allocation hot paths</li>
 *   <li><b>Flexibility:</b> Per-feature embedding dimensions for optimal representation</li>
 * </ul>
 * 
 * <p><b>Usage Example - RTB with Auto-Scaling:</b>
 * <pre>{@code
 * // Configure features for advertising model with auto-scaling
 * Layer.Spec inputLayer = Layers.inputMixed(optimizer,
 *     Feature.embedding(100000, 64),    // input[0]: bundle_id (100k bundles → 64-dim vectors)
 *     Feature.embedding(50000, 32),     // input[1]: publisher_id (50k publishers → 32-dim vectors)  
 *     Feature.oneHot(4),                // input[2]: connection_type (wifi/4g/3g/other → 4-dim one-hot)
 *     Feature.oneHot(8),                // input[3]: device_type (8 device types → 8-dim one-hot)
 *     Feature.autoScale(),              // input[4]: bid_floor ($0.01-$50.00 → auto-scaled to [0,1])
 *     Feature.autoNormalize(),          // input[5]: user_age (18-65 → auto-normalized mean=0, std=1)
 *     Feature.passthrough()             // input[6]: ctr (0.0-1.0 → already normalized, pass-through)
 * );
 * 
 * // Input data must match feature configuration order exactly
 * float[] rtbInput = {12345, 6789, 2, 5, 2.50f, 35.0f, 0.025f}; // bundle, pub, conn, device, bid_floor, age, ctr
 * LayerContext context = layer.forward(rtbInput);
 * float[] output = context.outputs(); // [64 + 32 + 4 + 8 + 1 + 1 + 1] = 111-dimensional feature vector
 * }</pre>
 */
public class MixedFeatureInputLayer implements Layer, Serializable {
    
    private final Optimizer optimizer;
    private final Feature[] features;
    private final float[][][] embeddings; // [featureIndex][valueIndex][embeddingDim] - only for embedding features
    private final int totalOutputDimension;
    // Instance buffer pool for output dimension arrays
    private final PooledFloatArray outputBufferPool;
    // Note: embeddingGradientBuffers needs to remain ThreadLocal as it's accumulated state
    private final ThreadLocal<float[][][]> embeddingGradientBuffers;
    
    // Scaling statistics for AUTO_NORMALIZE features
    private final float[] featureMean;    // For AUTO_NORMALIZE: running mean values
    private final float[] featureVariance; // For AUTO_NORMALIZE: running variance values
    private final long[] featureCount;    // Sample count for each feature
    
    /**
     * Create a mixed feature input layer.
     * 
     * @param optimizer optimizer for training embedding tables
     * @param features array of feature configurations, one per input position
     * @param initStrategy weight initialization strategy for embeddings
     */
    public MixedFeatureInputLayer(Optimizer optimizer, Feature[] features, WeightInitStrategy initStrategy) {
        // Comprehensive feature configuration validation
        if (features == null)
            throw new IllegalArgumentException("Features array cannot be null");
            
        if (features.length == 0)
            throw new IllegalArgumentException(
                "At least one feature must be configured. " +
                "Use Feature.embedding(), Feature.oneHot(), or Feature.passthrough() to define features.");
        
        // Validate all features are non-null and have reasonable configurations
        for (int i = 0; i < features.length; i++) {
            Feature feature = features[i];
            if (feature == null)
                throw new IllegalArgumentException(String.format(
                    "Feature %d is null. All features must be configured with valid Feature objects.", i));
                    
            // Check for reasonable embedding dimensions
            if (feature.getType() == Feature.Type.EMBEDDING) {
                if (feature.getEmbeddingDimension() > 1024)
                    System.err.println(String.format(
                        "Warning: Feature %d has embedding dimension %d, which is quite large. " +
                        "Consider using smaller dimensions (32-128) for better performance and memory usage.",
                        i, feature.getEmbeddingDimension()));
                        
                if (feature.getMaxUniqueValues() > 10_000_000)
                    System.err.println(String.format(
                        "Warning: Feature %d has vocabulary size %d, which is very large. " +
                        "This will use significant memory (%d MB). Consider using smaller vocabularies or feature hashing.",
                        i, feature.getMaxUniqueValues(), 
                        (feature.getMaxUniqueValues() * feature.getEmbeddingDimension() * 4) / (1024 * 1024)));
            }
            
            // Check for reasonable one-hot dimensions
            if (feature.getType() == Feature.Type.ONEHOT && feature.getMaxUniqueValues() > 1000)
                System.err.println(String.format(
                    "Warning: Feature %d has %d categories for one-hot encoding. " +
                    "Consider using Feature.embedding() instead for high-cardinality features (>100 categories).",
                    i, feature.getMaxUniqueValues()));
        }
        
        this.optimizer = optimizer;
        this.features = features.clone(); // Defensive copy
        this.totalOutputDimension = calculateTotalOutputDimension(features);
        
        // Validate feature naming - must be all named or none named
        validateFeatureNaming(features);
        
        // Initialize embedding tables only for embedding features
        this.embeddings = new float[features.length][][];
        for (int i = 0; i < features.length; i++) {
            if (features[i].getType() == Feature.Type.EMBEDDING || features[i].getType() == Feature.Type.HASHED_EMBEDDING) {
                int maxValues = features[i].getMaxUniqueValues();
                int embeddingDim = features[i].getEmbeddingDimension();
                this.embeddings[i] = new float[maxValues][embeddingDim];
                
                // Initialize embeddings with chosen strategy
                switch (initStrategy) {
                    case XAVIER -> NetMath.weightInitXavier(embeddings[i], embeddingDim, embeddingDim);
                    case HE -> NetMath.weightInitHe(embeddings[i], embeddingDim);
                }
            }
        }
        
        // Initialize scaling statistics arrays for AUTO_NORMALIZE features
        this.featureMean = new float[features.length];
        this.featureVariance = new float[features.length];
        this.featureCount = new long[features.length];
        
        // Initialize scaling arrays with appropriate defaults
        for (int i = 0; i < features.length; i++) {
            featureMean[i] = 0.0f;
            featureVariance[i] = 0.0f;
            featureCount[i] = 0;
        }
        
        // Initialize buffer pool for outputs
        this.outputBufferPool = new PooledFloatArray(totalOutputDimension);
        // ThreadLocal for gradient accumulation (required for thread-specific state)
        this.embeddingGradientBuffers = ThreadLocal.withInitial(() -> {
            float[][][] buffers = new float[features.length][][];
            for (int i = 0; i < features.length; i++) {
                if (features[i].getType() == Feature.Type.EMBEDDING || features[i].getType() == Feature.Type.HASHED_EMBEDDING) {
                    int maxValues = features[i].getMaxUniqueValues();
                    int embeddingDim = features[i].getEmbeddingDimension();
                    buffers[i] = new float[maxValues][embeddingDim];
                }
            }
            return buffers;
        });
    }
    
    private static int calculateTotalOutputDimension(Feature[] features) {
        int total = 0;
        for (Feature feature : features) {
            total += feature.getOutputDimension();
        }
        return total;
    }
    
    @Override
    public LayerContext forward(float[] input) {
        // Comprehensive input validation with helpful error messages
        if (input == null)
            throw new IllegalArgumentException("Input array cannot be null");
            
        if (input.length != features.length)
            throw new IllegalArgumentException(String.format(
                "Input array has %d elements but %d features were configured. " + 
                "Each input element must correspond to exactly one feature configuration. " +
                "Expected input format: [feature0_value, feature1_value, ..., feature%d_value]", 
                input.length, features.length, features.length - 1));
        
        float[] output = outputBufferPool.getBuffer();
        int outputPosition = 0;
        
        try {
            // Process each feature according to its configuration
        for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
            Feature feature = features[featureIndex];
            int inputValue = (int) input[featureIndex];
            
            // Validate that input is actually an integer for embedding/onehot features
            if ((feature.getType() == Feature.Type.EMBEDDING || feature.getType() == Feature.Type.ONEHOT) 
                && inputValue != input[featureIndex])
                throw new IllegalArgumentException(String.format(
                    "Feature %d (%s): input must be integer, got %.2f. " +
                    "Embedding and OneHot features require integer values representing category/token IDs. " +
                    "Use Feature.passthrough(), Feature.autoScale(minBound, maxBound), or Feature.autoNormalize() for continuous numerical values.", 
                    featureIndex, feature.getType(), input[featureIndex]));
            
            switch (feature.getType()) {
                case EMBEDDING -> {
                    // High-cardinality feature: lookup dense embedding
                    int maxValues = feature.getMaxUniqueValues();
                    if (inputValue < 0 || inputValue >= maxValues)
                        throw new IllegalArgumentException(String.format(
                            "Feature %d (embedding): value %d is out of range [0, %d). " +
                            "Embedding features expect token/category IDs from 0 to maxUniqueValues-1. " +
                            "Check your data preprocessing or increase maxUniqueValues to %d.", 
                            featureIndex, inputValue, maxValues, inputValue + 1));
                    
                    int embeddingDim = feature.getEmbeddingDimension();
                    System.arraycopy(embeddings[featureIndex][inputValue], 0, output, outputPosition, embeddingDim);
                    outputPosition += embeddingDim;
                }
                case ONEHOT -> {
                    // Low-cardinality feature: one-hot encoding
                    int numCategories = feature.getMaxUniqueValues(); // Stored in maxUniqueValues field
                    if (inputValue < 0 || inputValue >= numCategories)
                        throw new IllegalArgumentException(String.format(
                            "Feature %d (oneHot): value %d is out of range [0, %d). " +
                            "OneHot features expect category IDs from 0 to numberOfCategories-1. " +
                            "Check your data preprocessing or increase numberOfCategories to %d.", 
                            featureIndex, inputValue, numCategories, inputValue + 1));
                    
                    // Clear one-hot section
                    for (int j = 0; j < numCategories; j++) {
                        output[outputPosition + j] = 0.0f;
                    }
                    // Set the active category
                    output[outputPosition + inputValue] = 1.0f;
                    outputPosition += numCategories;
                }
                case PASSTHROUGH -> {
                    // Numerical feature: direct pass-through
                    output[outputPosition] = input[featureIndex];
                    outputPosition++;
                }
                case SCALE_BOUNDED -> {
                    // Numerical feature: min-max scaling with user-specified bounds
                    float value = input[featureIndex];
                    float scaledValue = applyBoundedScaling(featureIndex, value, feature);
                    output[outputPosition] = scaledValue;
                    outputPosition++;
                }
                case AUTO_NORMALIZE -> {
                    // Numerical feature: automatic z-score normalization (mean=0, std=1)
                    float value = input[featureIndex];
                    updateMeanVarianceStatistics(featureIndex, value);
                    float normalizedValue = applyZScoreNormalization(featureIndex, value);
                    output[outputPosition] = normalizedValue;
                    outputPosition++;
                }
                case HASHED_EMBEDDING -> {
                    // Hashed embedding feature: use multiple hash functions
                    // Input is expected to be a hash code (integer representation of string)
                    int hashCode = (int) input[featureIndex];
                    int embeddingDim = feature.getEmbeddingDimension();
                    int hashBuckets = feature.getMaxUniqueValues();
                    
                    // Compute 3 hash positions
                    int[] positions = computeHashPositions(hashCode, hashBuckets);
                    
                    // Average embeddings from all positions
                    for (int d = 0; d < embeddingDim; d++)
                        output[outputPosition + d] = 0.0f;
                    
                    for (int pos : positions) {
                        for (int d = 0; d < embeddingDim; d++)
                            output[outputPosition + d] += embeddings[featureIndex][pos][d];
                    }
                    
                    // Scale by 1/3 to get average
                    float scale = 1.0f / 3.0f;
                    for (int d = 0; d < embeddingDim; d++)
                        output[outputPosition + d] *= scale;
                    
                    outputPosition += embeddingDim;
                }
            }
            }
            
            // Create result array with exactly the right size
            float[] result = new float[totalOutputDimension];
            System.arraycopy(output, 0, result, 0, totalOutputDimension);
            return new LayerContext(input, null, result);
        } finally {
            outputBufferPool.releaseBuffer(output);
        }
    }
    
    @Override
    public LayerContext forward(float[] input, ExecutorService executor) {
        // Feature lookups and one-hot encoding are not parallelizable due to simple operations
        return forward(input);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient, ExecutorService executor) {
        LayerContext context = stack[stackIndex];
        float[] inputFloats = context.inputs();
        
        if (upstreamGradient.length != totalOutputDimension)
            throw new IllegalArgumentException(String.format(
                "Gradient length mismatch: expected %d, got %d", 
                totalOutputDimension, upstreamGradient.length));
        
        float[][][] embeddingGradients = embeddingGradientBuffers.get();
        
        // Clear gradients for embedding features
        for (int i = 0; i < features.length; i++) {
            if (features[i].getType() == Feature.Type.EMBEDDING || features[i].getType() == Feature.Type.HASHED_EMBEDDING) {
                int maxValues = features[i].getMaxUniqueValues();
                int embeddingDim = features[i].getEmbeddingDimension();
                for (int j = 0; j < maxValues; j++) {
                    for (int k = 0; k < embeddingDim; k++) {
                        embeddingGradients[i][j][k] = 0.0f;
                    }
                }
            }
        }
        
        // Accumulate gradients from upstream
        int gradientPosition = 0;
        for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
            Feature feature = features[featureIndex];
            
            switch (feature.getType()) {
                case EMBEDDING -> {
                    // Accumulate embedding gradients
                    int inputValue = (int) inputFloats[featureIndex];
                    int embeddingDim = feature.getEmbeddingDimension();
                    
                    for (int j = 0; j < embeddingDim; j++) {
                        embeddingGradients[featureIndex][inputValue][j] += upstreamGradient[gradientPosition + j];
                    }
                    gradientPosition += embeddingDim;
                }
                case ONEHOT -> {
                    // One-hot features don't have learnable parameters, skip gradients
                    int numCategories = feature.getMaxUniqueValues();
                    gradientPosition += numCategories;
                }
                case PASSTHROUGH, AUTO_NORMALIZE, SCALE_BOUNDED -> {
                    // These features don't have learnable parameters, skip gradients
                    gradientPosition++;
                }
                case HASHED_EMBEDDING -> {
                    // Accumulate gradients for hashed embeddings
                    int hashCode = (int) inputFloats[featureIndex];
                    int embeddingDim = feature.getEmbeddingDimension();
                    int hashBuckets = feature.getMaxUniqueValues();
                    
                    // Get hash positions
                    int[] positions = computeHashPositions(hashCode, hashBuckets);
                    
                    // Debug logging
                    if (Math.random() < 0.01) { // Log 1% of the time
                        float gradNorm = 0;
                        for (int j = 0; j < embeddingDim; j++) {
                            gradNorm += upstreamGradient[gradientPosition + j] * upstreamGradient[gradientPosition + j];
                        }
                        gradNorm = (float) Math.sqrt(gradNorm);
                        System.out.println("DEBUG: Hashed embedding upstream gradient norm: " + gradNorm + 
                                         " for feature " + featureIndex + " positions: " + Arrays.toString(positions));
                    }
                    
                    // Accumulate gradients for each position (scaled by 1/3)
                    float scale = 1.0f / 3.0f;
                    for (int pos : positions) {
                        for (int j = 0; j < embeddingDim; j++) {
                            embeddingGradients[featureIndex][pos][j] += upstreamGradient[gradientPosition + j] * scale;
                        }
                    }
                    gradientPosition += embeddingDim;
                }
            }
        }
        
        // Update embeddings using optimizer
        for (int i = 0; i < features.length; i++) {
            if (features[i].getType() == Feature.Type.EMBEDDING || features[i].getType() == Feature.Type.HASHED_EMBEDDING) {
                optimizer.optimize(embeddings[i], new float[0], embeddingGradients[i], new float[0]);
                
                // Clear gradients after update - CRITICAL!
                for (int j = 0; j < embeddingGradients[i].length; j++) {
                    Arrays.fill(embeddingGradients[i][j], 0.0f);
                }
            }
        }
        
        return null; // Input layer doesn't propagate gradients further
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        return backward(stack, stackIndex, upstreamGradient, (ExecutorService) null);
    }
    
    @Override
    public int getOutputSize() {
        return totalOutputDimension;
    }
    
    
    /**
     * Get the embedding vector for a specific embedding feature and value.
     * 
     * @param featureIndex index of the feature (must be an embedding feature)
     * @param value the input value to get the embedding for
     * @return copy of the embedding vector
     */
    public float[] getEmbedding(int featureIndex, int value) {
        if (featureIndex < 0 || featureIndex >= features.length)
            throw new IllegalArgumentException("Invalid feature index: " + featureIndex);
        if (features[featureIndex].getType() != Feature.Type.EMBEDDING)
            throw new IllegalArgumentException("Feature " + featureIndex + " is not an embedding feature");
        
        int maxValues = features[featureIndex].getMaxUniqueValues();
        if (value < 0 || value >= maxValues)
            throw new IllegalArgumentException("Invalid value: " + value);
        
        return embeddings[featureIndex][value].clone();
    }
    
    /**
     * Get the feature configurations for this layer.
     * @return defensive copy of feature array
     */
    public Feature[] getFeatures() {
        return features.clone();
    }
    
    /**
     * Get the feature names from this layer if they were specified.
     * @return array of feature names (may contain nulls for unnamed features)
     */
    public String[] getFeatureNames() {
        String[] names = new String[features.length];
        for (int i = 0; i < features.length; i++) {
            names[i] = features[i].getName();
        }
        return names;
    }
    
    /**
     * Check if all features have explicit names.
     * @return true if all features have non-null names
     */
    public boolean hasExplicitFeatureNames() {
        for (Feature feature : features) {
            if (feature.getName() == null) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Validate that features are either ALL named or NONE named.
     * Mixed naming (some named, some not) is not allowed as it creates confusion.
     * 
     * @param features array of features to validate
     * @throws IllegalArgumentException if features have mixed naming
     */
    private static void validateFeatureNaming(Feature[] features) {
        int namedCount = 0;
        int unnamedCount = 0;
        List<Integer> unnamedIndices = new ArrayList<>();
        
        for (int i = 0; i < features.length; i++) {
            if (features[i].getName() != null) {
                namedCount++;
            } else {
                unnamedCount++;
                unnamedIndices.add(i);
            }
        }
        
        // Check for mixed naming
        if (namedCount > 0 && unnamedCount > 0) {
            StringBuilder message = new StringBuilder();
            message.append("Feature naming must be all-or-nothing. Found ")
                   .append(namedCount).append(" named and ")
                   .append(unnamedCount).append(" unnamed features.\n");
            message.append("Unnamed features at indices: ").append(unnamedIndices).append("\n");
            message.append("Either:\n");
            message.append("1. Name ALL features (recommended for Map-based inputs):\n");
            
            for (int idx : unnamedIndices) {
                Feature f = features[idx];
                message.append("   Feature[").append(idx).append("]: ");
                switch (f.getType()) {
                    case EMBEDDING:
                        message.append("Feature.embedding(").append(f.getMaxUniqueValues())
                               .append(", ").append(f.getEmbeddingDimension())
                               .append(", \"your_feature_name\")\n");
                        break;
                    case ONEHOT:
                        message.append("Feature.oneHot(").append(f.getMaxUniqueValues())
                               .append(", \"your_feature_name\")\n");
                        break;
                    case PASSTHROUGH:
                        message.append("Feature.passthrough(\"your_feature_name\")\n");
                        break;
                    case AUTO_NORMALIZE:
                        message.append("Feature.autoNormalize(\"your_feature_name\")\n");
                        break;
                    case SCALE_BOUNDED:
                        message.append("Feature.autoScale(").append(f.getMinBound())
                               .append("f, ").append(f.getMaxBound())
                               .append("f, \"your_feature_name\")\n");
                        break;
                }
            }
            
            message.append("2. Or use NO names (for array-based inputs only):\n");
            message.append("   Remove names from all features that currently have them.");
            
            throw new IllegalArgumentException(message.toString());
        }
    }
    
    /**
     * Create a layer specification for a mixed feature input layer.
     */
    public static Layer.Spec spec(Optimizer optimizer, Feature[] features, WeightInitStrategy initStrategy) {
        return new MixedFeatureInputLayerSpec(optimizer, features, initStrategy, 1.0);
    }
    
    /**
     * Create a mixed feature layer specification with custom learning rate ratio.
     */
    public static Layer.Spec spec(Optimizer optimizer, Feature[] features, 
                                  WeightInitStrategy initStrategy, double learningRateRatio) {
        return new MixedFeatureInputLayerSpec(optimizer, features, initStrategy, learningRateRatio);
    }
    
    /**
     * Specification for creating mixed feature layers with optimizer management.
     */
    static class MixedFeatureInputLayerSpec extends BaseLayerSpec<MixedFeatureInputLayerSpec> {
        private final Feature[] features;
        private final WeightInitStrategy initStrategy;
        
        public MixedFeatureInputLayerSpec(Optimizer optimizer, Feature[] features, 
                                          WeightInitStrategy initStrategy, double learningRateRatio) {
            super(calculateTotalOutputDimension(features), optimizer);
            this.features = features.clone(); // Defensive copy
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        @Override
        public Layer create(int inputSize) {
            return createLayer(inputSize, getEffectiveOptimizer(null));
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            // Input size is ignored for mixed feature layers (determined by feature configuration)
            return new MixedFeatureInputLayer(effectiveOptimizer, features, initStrategy);
        }
    }
    
    // ===============================
    // FEATURE SCALING HELPER METHODS
    // ===============================
    
    /**
     * Apply bounded min-max scaling using user-specified bounds.
     * Values are scaled to [0,1] range and clamped if outside bounds.
     */
    private float applyBoundedScaling(int featureIndex, float value, Feature feature) {
        float minBound = feature.getMinBound();
        float maxBound = feature.getMaxBound();
        
        // Scale to [0,1] using user-specified bounds
        float scaled = (value - minBound) / (maxBound - minBound);
        
        // Clamp to [0,1] range for values outside expected bounds
        return Math.max(0.0f, Math.min(1.0f, scaled));
    }
    
    /**
     * Update running mean and variance statistics for AUTO_NORMALIZE features.
     * Uses Welford's online algorithm for numerical stability.
     * Thread-safe with synchronized updates.
     */
    private synchronized void updateMeanVarianceStatistics(int featureIndex, float value) {
        long n = featureCount[featureIndex] + 1;
        featureCount[featureIndex] = n;
        
        // Welford's online algorithm for mean and variance
        float delta = value - featureMean[featureIndex];
        featureMean[featureIndex] += delta / n;
        float delta2 = value - featureMean[featureIndex];
        featureVariance[featureIndex] += delta * delta2;
    }
    
    /**
     * Apply z-score normalization to transform value to mean=0, std=1.
     */
    private float applyZScoreNormalization(int featureIndex, float value) {
        if (featureCount[featureIndex] < 2) {
            return 0.0f; // Not enough data to normalize
        }
        
        float mean = featureMean[featureIndex];
        float variance = featureVariance[featureIndex] / (featureCount[featureIndex] - 1);
        float std = (float) Math.sqrt(variance);
        
        // Handle edge case where std == 0 (constant feature)
        if (std < 1e-8f) {
            return 0.0f;
        }
        
        return (value - mean) / std;
    }
    
    /**
     * Compute hash positions using multiple hash functions for HASHED_EMBEDDING.
     * Uses MurmurHash3-inspired mixing for better distribution.
     */
    private int[] computeHashPositions(int hashCode, int hashBuckets) {
        // MurmurHash3 prime seeds for better distribution
        final int[] HASH_SEEDS = {
            0x1b873593,  // MurmurHash3 constant c1
            0xcc9e2d51,  // MurmurHash3 constant c2
            0x85ebca6b   // MurmurHash3 mix constant
        };
        
        int[] positions = new int[3];
        
        for (int i = 0; i < 3; i++) {
            // MurmurHash3-inspired mixing
            int h = hashCode;
            h ^= h >>> 16;
            h *= HASH_SEEDS[i];
            h ^= h >>> 13;
            h *= 0x5bd1e995;
            h ^= h >>> 15;
            
            // Map to bucket (ensure positive)
            positions[i] = Math.abs(h) % hashBuckets;
        }
        
        return positions;
    }
    
    // ===============================
    // FEATURE SCALING UTILITIES
    // ===============================
    
    /**
     * Get scaling statistics for a feature (useful for debugging/monitoring).
     * 
     * @param featureIndex index of the feature
     * @return map with scaling statistics (min, max, mean, std, count)
     */
    public Map<String, Number> getFeatureStatistics(int featureIndex) {
        if (featureIndex < 0 || featureIndex >= features.length) {
            throw new IllegalArgumentException("Feature index out of range: " + featureIndex);
        }
        
        Feature.Type type = features[featureIndex].getType();
        Map<String, Number> stats = new java.util.HashMap<>();
        stats.put("type", type.ordinal());
        stats.put("count", featureCount[featureIndex]);
        
        if (type == Feature.Type.AUTO_NORMALIZE) {
            stats.put("mean", featureMean[featureIndex]);
            if (featureCount[featureIndex] > 1) {
                float variance = featureVariance[featureIndex] / (featureCount[featureIndex] - 1);
                stats.put("variance", variance);
                stats.put("std", Math.sqrt(variance));
            }
        }
        
        return stats;
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        // Write number of features
        out.writeInt(features.length);
        
        // Write feature configurations
        for (int i = 0; i < features.length; i++) {
            Feature feature = features[i];
            out.writeInt(feature.getType().ordinal());
            switch (feature.getType()) {
                case EMBEDDING -> {
                    out.writeInt(feature.getMaxUniqueValues());
                    out.writeInt(feature.getEmbeddingDimension());
                }
                case ONEHOT -> {
                    out.writeInt(feature.getMaxUniqueValues()); // numberOfCategories
                }
                case PASSTHROUGH -> {
                    // No parameters to serialize
                }
                case SCALE_BOUNDED -> {
                    // Serialize user-specified bounds
                    out.writeFloat(feature.getMinBound());
                    out.writeFloat(feature.getMaxBound());
                }
                case AUTO_NORMALIZE -> {
                    // Serialize mean/variance statistics
                    out.writeFloat(featureMean[i]);
                    out.writeFloat(featureVariance[i]);
                    out.writeLong(featureCount[i]);
                }
            }
        }
        
        // Write embeddings for embedding features
        for (int i = 0; i < features.length; i++) {
            if (features[i].getType() == Feature.Type.EMBEDDING) {
                int maxValues = features[i].getMaxUniqueValues();
                int embeddingDim = features[i].getEmbeddingDimension();
                for (int j = 0; j < maxValues; j++) {
                    for (int k = 0; k < embeddingDim; k++) {
                        out.writeFloat(embeddings[i][j][k]);
                    }
                }
            }
        }
        
        // Write optimizer using centralized service
        Integer typeId = SerializationService.getTypeId(optimizer);
        if (typeId == null) {
            // Fallback to direct serialization if not registered
            Serializable serializableOptimizer = (Serializable) optimizer;
            out.writeInt(serializableOptimizer.getTypeId());
            serializableOptimizer.writeTo(out, version);
        } else {
            SerializationService.writeWithTypeId(out, (Serializable) optimizer, version);
        }
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize a MixedFeatureInputLayer from stream.
     */
    public static MixedFeatureInputLayer deserialize(DataInputStream in, int version) throws IOException {
        // Read number of features
        int numFeatures = in.readInt();
        
        // Read feature configurations
        Feature[] features = new Feature[numFeatures];
        float[] storedMeans = new float[numFeatures];
        float[] storedVariances = new float[numFeatures];
        long[] storedCounts = new long[numFeatures];
        
        for (int i = 0; i < numFeatures; i++) {
            int typeOrdinal = in.readInt();
            Feature.Type type = Feature.Type.values()[typeOrdinal];
            
            switch (type) {
                case EMBEDDING -> {
                    int maxUniqueValues = in.readInt();
                    int embeddingDimension = in.readInt();
                    features[i] = Feature.embedding(maxUniqueValues, embeddingDimension);
                }
                case ONEHOT -> {
                    int numberOfCategories = in.readInt();
                    features[i] = Feature.oneHot(numberOfCategories);
                }
                case PASSTHROUGH -> {
                    features[i] = Feature.passthrough();
                }
                case SCALE_BOUNDED -> {
                    float minBound = in.readFloat();
                    float maxBound = in.readFloat();
                    features[i] = Feature.autoScale(minBound, maxBound);
                }
                case AUTO_NORMALIZE -> {
                    // Read the statistics from the stream (they were written during feature config)
                    storedMeans[i] = in.readFloat();
                    storedVariances[i] = in.readFloat();
                    storedCounts[i] = in.readLong();
                    features[i] = Feature.autoNormalize();
                }
            }
        }
        
        // Read embeddings (to match write order: features, embeddings, optimizer)
        float[][][] embeddingData = new float[numFeatures][][];
        for (int i = 0; i < numFeatures; i++) {
            if (features[i].getType() == Feature.Type.EMBEDDING) {
                int maxValues = features[i].getMaxUniqueValues();
                int embeddingDim = features[i].getEmbeddingDimension();
                embeddingData[i] = new float[maxValues][embeddingDim];
                for (int j = 0; j < maxValues; j++) {
                    for (int k = 0; k < embeddingDim; k++) {
                        embeddingData[i][j][k] = in.readFloat();
                    }
                }
            }
        }
        
        // Now read optimizer
        int optimizerTypeId = in.readInt();
        Optimizer optimizer = SerializationService.deserializeOptimizer(in, optimizerTypeId, version);
        
        // Create layer
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        
        // Copy embedding data to layer
        for (int i = 0; i < numFeatures; i++) {
            if (features[i].getType() == Feature.Type.EMBEDDING && embeddingData[i] != null) {
                layer.embeddings[i] = embeddingData[i];
            }
        }
        
        // Copy stored AUTO_NORMALIZE statistics to layer (they were read during feature config)
        for (int i = 0; i < numFeatures; i++) {
            if (features[i].getType() == Feature.Type.AUTO_NORMALIZE) {
                layer.featureMean[i] = storedMeans[i];
                layer.featureVariance[i] = storedVariances[i];
                layer.featureCount[i] = storedCounts[i];
            }
        }
        
        return layer;
    }
    
    private static void writeOptimizer(DataOutputStream out, Optimizer optimizer, int version) throws IOException {
        System.out.println("[DEBUG] Writing optimizer: " + optimizer.getClass().getSimpleName());
        
        String registeredName = SerializationRegistry.getRegisteredName(optimizer);
        System.out.println("[DEBUG] Registered name: " + registeredName);
        
        if (registeredName != null) {
            System.out.println("[DEBUG] Using custom serialization for: " + registeredName);
            out.writeInt(SerializationConstants.TYPE_CUSTOM);
            out.writeUTF(registeredName);
            return;
        }
        
        // Fall back to built-in serialization
        Serializable serializableOptimizer = (Serializable) optimizer;
        int typeId = serializableOptimizer.getTypeId();
        System.out.println("[DEBUG] Using built-in serialization, type ID: " + typeId);
        out.writeInt(typeId);
        serializableOptimizer.writeTo(out, version);
    }
    
    private static Optimizer readOptimizer(DataInputStream in, int version) throws IOException {
        int typeId = in.readInt();
        System.out.println("[DEBUG] Reading optimizer with type ID: " + typeId);
        
        if (typeId == SerializationConstants.TYPE_CUSTOM) {
            String className = in.readUTF();
            System.out.println("[DEBUG] Reading custom optimizer: " + className);
            return SerializationRegistry.createOptimizer(className, in, version);
        }
        
        System.out.println("[DEBUG] Reading built-in optimizer with type ID: " + typeId);
        System.out.println("[DEBUG] Expected type IDs - SGD: " + SerializationConstants.TYPE_SGD_OPTIMIZER + 
                          ", ADAM: " + SerializationConstants.TYPE_ADAM_OPTIMIZER + 
                          ", ADAMW: " + SerializationConstants.TYPE_ADAMW_OPTIMIZER);
        
        return switch (typeId) {
            case SerializationConstants.TYPE_SGD_OPTIMIZER -> SgdOptimizer.deserialize(in, version);
            case SerializationConstants.TYPE_ADAM_OPTIMIZER -> AdamOptimizer.deserialize(in, version);
            case SerializationConstants.TYPE_ADAMW_OPTIMIZER -> AdamWOptimizer.deserialize(in, version);
            default -> throw new IOException("Unknown optimizer type ID: " + typeId + 
                " (expected SGD=" + SerializationConstants.TYPE_SGD_OPTIMIZER + 
                ", ADAM=" + SerializationConstants.TYPE_ADAM_OPTIMIZER + 
                ", ADAMW=" + SerializationConstants.TYPE_ADAMW_OPTIMIZER + ")");
        };
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 4; // numFeatures
        
        // Feature configurations
        for (Feature feature : features) {
            size += 4; // type ordinal
            switch (feature.getType()) {
                case EMBEDDING -> size += 8; // maxUniqueValues + embeddingDimension
                case ONEHOT -> size += 4; // numberOfCategories
                case PASSTHROUGH -> { /* no parameters */ }
                case AUTO_NORMALIZE -> size += 12; // mean + variance + count
                case SCALE_BOUNDED -> size += 8; // minBound + maxBound
            }
        }
        
        // Embeddings
        for (Feature feature : features) {
            if (feature.getType() == Feature.Type.EMBEDDING) {
                size += feature.getMaxUniqueValues() * feature.getEmbeddingDimension() * 4; // float values
            }
        }
        
        // Optimizer size using centralized service
        size += 4; // type ID
        size += ((Serializable) optimizer).getSerializedSize(version); // optimizer data
        
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_MIXED_FEATURE_INPUT_LAYER;
    }
}