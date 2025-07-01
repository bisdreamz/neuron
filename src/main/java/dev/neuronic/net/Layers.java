package dev.neuronic.net;

import dev.neuronic.net.activators.*;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.layers.DropoutLayer;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.GruLayer;
import dev.neuronic.net.layers.InputEmbeddingLayer;
import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.LayerNormLayer;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.outputs.*;

/**
 * Convenient factory for all layer types.
 * 
 * Organized by purpose for easy discovery:
 * - hiddenXxx() - Hidden/intermediate layers
 * - outputXxx() - Terminal layers that handle loss computation
 * 
 * Usage:
 * NeuralNet.newBuilder()
 *   .input(784)
 *   .layer(Layers.hiddenDense(256, Activators.relu(), optimizer))
 *   .layer(Layers.hiddenDense(128, Activators.relu(), optimizer))
 *   .output(Layers.outputSoftmaxCrossEntropy(10, optimizer))
 *   .build()
 */
public final class Layers {
    
    private Layers() {} // Utility class
    
    // ===============================
    // INPUT LAYERS
    // ===============================
    
    /**
     * Input embedding layer for converting discrete tokens to dense vectors.
     * 
     * <p><b>What it does:</b> Transforms token IDs (words, items, categories) into learnable 
     * dense vector representations that capture semantic meaning.
     * 
     * <p><b>When to use:</b>
     * <ul>
     *   <li><b>Natural Language Processing:</b> Word embeddings for text models (BERT, GPT)</li>
     *   <li><b>Recommendation systems:</b> User/item embeddings for collaborative filtering</li>
     *   <li><b>Categorical features:</b> Converting high-cardinality categories to dense vectors</li>
     *   <li><b>Sequence modeling:</b> Any discrete sequence data (DNA, music, time series categories)</li>
     * </ul>
     * 
     * <p><b>Key benefits:</b>
     * <ul>
     *   <li><b>Semantic learning:</b> Similar tokens learn similar embeddings automatically</li>
     *   <li><b>Dimensionality control:</b> Convert sparse one-hot to dense fixed-size vectors</li>
     *   <li><b>Memory efficient:</b> Shared representations for repeated tokens</li>
     *   <li><b>Transfer learning:</b> Pre-trained embeddings can be fine-tuned</li>
     * </ul>
     * 
     * <p><b>Example Usage:</b>
     * <pre>{@code
     * // Language model with 30,000 word vocabulary → 300-dimensional embeddings
     * NeuralNet textModel = NeuralNet.newBuilder()
     *     .input(sequenceLength)  // e.g., 50 tokens per sentence
     *     .layer(Layers.inputEmbedding(30000, 300, optimizer))
     *     .layer(Layers.hiddenDenseRelu(512))
     *     .output(Layers.outputSoftmaxCrossEntropy(numClasses));
     * 
     * // Input: token IDs [42, 1337, 256] 
     * // Output: [3 × 300] = 900-dimensional concatenated embeddings
     * }</pre>
     * 
     * <p><b>Parameter Guidelines:</b>
     * <ul>
     *   <li><b>vocabSize:</b> Number of unique tokens in your dataset (e.g., 30k words, 1M items)</li>
     *   <li><b>embeddingDim:</b> 50-300 for small vocabs, 300-1024 for large vocabs</li>
     *   <li><b>Memory:</b> vocabSize × embeddingDim × 4 bytes (e.g., 30k × 300 = 36MB)</li>
     * </ul>
     * 
     * @param vocabSize vocabulary size - must match your token ID range [0, vocabSize-1]
     * @param embeddingDim embedding vector dimension - higher = more expressive but more memory
     * @param optimizer optimizer for learning embeddings during training
     * @return input embedding layer spec ready for neural network construction
     */
    public static Layer.Spec inputEmbedding(int vocabSize, int embeddingDim, Optimizer optimizer) {
        return InputEmbeddingLayer.spec(vocabSize, embeddingDim, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Input embedding layer with default network optimizer.
     * 
     * <p>Uses the neural network's default optimizer set with {@code .setDefaultOptimizer()}.
     * Convenient when all layers use the same optimizer settings.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * AdamOptimizer defaultOpt = new AdamOptimizer(0.001f);
     * NeuralNet model = NeuralNet.newBuilder()
     *     .input(sequenceLength)
     *     .setDefaultOptimizer(defaultOpt)  // All layers will use this
     *     .layer(Layers.inputEmbedding(50000, 256))  // Uses defaultOpt automatically
     *     .layer(Layers.hiddenDenseRelu(512))        // Uses defaultOpt automatically
     *     .output(Layers.outputSoftmaxCrossEntropy(10));
     * }</pre>
     * 
     * @param vocabSize vocabulary size - number of unique tokens/items
     * @param embeddingDim embedding vector dimension 
     * @return input embedding layer spec using network's default optimizer
     */
    public static Layer.Spec inputEmbedding(int vocabSize, int embeddingDim) {
        return InputEmbeddingLayer.spec(vocabSize, embeddingDim, null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Input embedding layer with custom weight initialization strategy.
     * 
     * <p>Advanced usage for fine-tuning embedding initialization. Most users should use 
     * the simpler methods above which default to Xavier initialization.
     * 
     * <p><b>Initialization strategies:</b>
     * <ul>
     *   <li><b>XAVIER:</b> Good general choice, helps with gradient flow</li>
     *   <li><b>HE:</b> Better for ReLU activations in subsequent layers</li>
     * </ul>
     * 
     * @param vocabSize vocabulary size 
     * @param embeddingDim embedding dimension
     * @param optimizer optimizer for embeddings
     * @param initStrategy weight initialization strategy (XAVIER or HE)
     * @return input embedding layer spec with custom initialization
     */
    public static Layer.Spec inputEmbedding(int vocabSize, int embeddingDim, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return InputEmbeddingLayer.spec(vocabSize, embeddingDim, optimizer, initStrategy);
    }
    
    /**
     * Input layer for SEQUENCES where all positions share the SAME vocabulary.
     * 
     * <p><b>⚠️ NOT THE SAME AS inputAllEmbeddings() ⚠️</b>
     * 
     * <p><b>Clear Distinction:</b>
     * <pre>
     * inputAllEmbeddings(256, opt, 10000, 5000, 2000):
     *   - Creates 3 DIFFERENT embedding tables
     *   - Feature 0: user_id (10000 possible IDs)
     *   - Feature 1: item_id (5000 possible IDs)  
     *   - Feature 2: category_id (2000 possible IDs)
     *   - Memory: (10000 + 5000 + 2000) × 256 × 4 bytes
     * 
     * inputSequenceEmbedding(35, 30000, 256, opt):
     *   - Creates 1 SHARED embedding table 
     *   - Position 0: word from 30k vocab
     *   - Position 1: word from SAME 30k vocab
     *   - Position 35: word from SAME 30k vocab
     *   - Memory: 30000 × 256 × 4 bytes only!
     * </pre>
     * 
     * <p><b>When to use each:</b>
     * <ul>
     *   <li><b>inputSequenceEmbedding:</b> Language models, DNA sequences, time series categories</li>
     *   <li><b>inputAllEmbeddings:</b> Ad features, user/item features, mixed ID types</li>
     *   <li><b>inputEmbedding:</b> Raw embedding layer when you handle tokenization yourself</li>
     * </ul>
     * 
     * <p><b>Example - Language Model:</b>
     * <pre>{@code
     * // WikiText-2 language model with 35-word context
     * NeuralNet model = NeuralNet.newBuilder()
     *     .input(35)  // sequence length
     *     .layer(Layers.inputSequenceEmbedding(35, 30000, 256, optimizer))
     *     .layer(Layers.gru(512))
     *     .output(Layers.outputSoftmaxCrossEntropy(30000));
     * 
     * SimpleNetString lm = SimpleNet.ofStringClassification(model);
     * 
     * // Train with string sequences
     * String[] context = {"The", "quick", "brown", "fox", ...};  // 35 words
     * lm.train(context, "jumps");  // Predict next word
     * }</pre>
     * 
     * @param sequenceLength number of positions in sequence (e.g., 35 for 35-word context)
     * @param sharedVocabSize vocabulary size shared across ALL positions (e.g., 30000 words)
     * @param embeddingDim embedding vector dimension (e.g., 256)
     * @param optimizer optimizer for learning embeddings
     * @return input sequence embedding layer for language modeling
     */
    public static Layer.Spec inputSequenceEmbedding(int sequenceLength, int sharedVocabSize, 
                                                    int embeddingDim, Optimizer optimizer) {
        return InputSequenceEmbeddingLayer.spec(sequenceLength, sharedVocabSize, embeddingDim, 
                                               optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Input sequence embedding layer with default optimizer.
     * 
     * <p>Uses the neural network's default optimizer set with {@code .setDefaultOptimizer()}.
     * 
     * @param sequenceLength number of positions in sequence
     * @param sharedVocabSize vocabulary size shared across ALL positions
     * @param embeddingDim embedding vector dimension
     * @return input sequence embedding layer using default optimizer
     */
    public static Layer.Spec inputSequenceEmbedding(int sequenceLength, int sharedVocabSize, 
                                                    int embeddingDim) {
        return InputSequenceEmbeddingLayer.spec(sequenceLength, sharedVocabSize, embeddingDim, 
                                               null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Input sequence embedding layer with custom initialization.
     * 
     * @param sequenceLength number of positions in sequence
     * @param sharedVocabSize vocabulary size shared across ALL positions
     * @param embeddingDim embedding vector dimension
     * @param optimizer optimizer for learning embeddings
     * @param initStrategy weight initialization strategy (XAVIER or HE)
     * @return input sequence embedding layer with custom initialization
     */
    public static Layer.Spec inputSequenceEmbedding(int sequenceLength, int sharedVocabSize, 
                                                    int embeddingDim, Optimizer optimizer,
                                                    WeightInitStrategy initStrategy) {
        return InputSequenceEmbeddingLayer.spec(sequenceLength, sharedVocabSize, embeddingDim, 
                                               optimizer, initStrategy);
    }
    
    /**
     * Mixed feature input layer for advertising and recommendation systems.
     * 
     * <p>Handles different feature types efficiently in a single layer:
     * <ul>
     *   <li><b>High-cardinality features:</b> Bundle IDs, publisher IDs → dense embeddings</li>
     *   <li><b>Low-cardinality features:</b> Connection type, device type → one-hot encoding</li>
     *   <li><b>Numerical features:</b> Age, price, duration → pass-through values</li>
     * </ul>
     * 
     * <p><b>Example for advertising:</b>
     * <pre>{@code
     * Layer.Spec inputLayer = Layers.inputMixed(optimizer,
     *     Feature.embedding(100000, 64),    // input[0]: bundle_id (100k bundles → 64-dim)
     *     Feature.embedding(50000, 32),     // input[1]: publisher_id (50k publishers → 32-dim)
     *     Feature.oneHot(4),                // input[2]: connection_type (wifi/4g/3g/other)
     *     Feature.oneHot(8),                // input[3]: device_type (phone/tablet/etc)
     *     Feature.passthrough()             // input[4]: user_age (continuous value)
     * );
     * 
     * // Input: [12345, 6789, 2, 5, 28.5] → Output: 109-dimensional feature vector
     * }</pre>
     * 
     * @param optimizer optimizer for training embedding tables
     * @param featureConfigurations feature configurations in input data order
     * @return mixed feature input layer spec
     */
    public static Layer.Spec inputMixed(Optimizer optimizer, Feature... featureConfigurations) {
        // Validate input to prevent common configuration errors
        if (featureConfigurations == null)
            throw new IllegalArgumentException("Feature configurations cannot be null");
        if (featureConfigurations.length == 0)
            throw new IllegalArgumentException(
                "At least one feature must be configured. " +
                "Example: Layers.inputMixed(optimizer, Feature.embedding(1000, 32), Feature.oneHot(4))");
        
        return MixedFeatureInputLayer.spec(optimizer, featureConfigurations, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Mixed feature input layer using default optimizer.
     * 
     * @param featureConfigurations feature configurations in input data order
     * @return mixed feature input layer spec
     */
    public static Layer.Spec inputMixed(Feature... featureConfigurations) {
        // Validate input to prevent common configuration errors
        if (featureConfigurations == null)
            throw new IllegalArgumentException("Feature configurations cannot be null");
        if (featureConfigurations.length == 0)
            throw new IllegalArgumentException(
                "At least one feature must be configured. " +
                "Example: Layers.inputMixed(Feature.embedding(1000, 32), Feature.oneHot(4))");
        
        return MixedFeatureInputLayer.spec(null, featureConfigurations, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Mixed feature input layer with custom initialization strategy.
     * 
     * @param optimizer optimizer for embedding tables
     * @param initStrategy weight initialization strategy for embeddings
     * @param featureConfigurations feature configurations in input data order
     * @return mixed feature input layer spec
     */
    public static Layer.Spec inputMixed(Optimizer optimizer, WeightInitStrategy initStrategy, Feature... featureConfigurations) {
        return MixedFeatureInputLayer.spec(optimizer, featureConfigurations, initStrategy);
    }
    
    /**
     * Multiple DIFFERENT embedding features with INDEPENDENT vocabularies.
     * 
     * <p><b>⚠️ NOT FOR SEQUENCES - Use inputSequenceEmbedding() for language models ⚠️</b>
     * 
     * <p>Creates SEPARATE embedding tables for each feature. Each position represents
     * a different feature type with its own vocabulary.
     * 
     * <p><b>Example - Ad Tech Features (DIFFERENT vocabularies):</b>
     * <pre>{@code
     * // 3 different ID types, each with own vocabulary
     * Layers.inputAllEmbeddings(64, optimizer, 
     *     100000,  // Feature 0: bundle_id (100k apps)
     *     50000,   // Feature 1: publisher_id (50k publishers)
     *     25000    // Feature 2: category_id (25k categories)
     * )
     * // Creates 3 independent embedding tables
     * // Total memory: (100k + 50k + 25k) × 64 × 4 bytes
     * }</pre>
     * 
     * <p><b>Wrong for language models!</b> Use inputSequenceEmbedding() when
     * positions share vocabulary (words in a sentence).
     * 
     * @param embeddingDimension size of embedding vectors for all features
     * @param optimizer optimizer for embedding tables
     * @param maxUniqueValuesPerFeature vocabulary size for each feature (in input order)
     * @return mixed feature input layer spec with all embedding features
     */
    public static Layer.Spec inputAllEmbeddings(int embeddingDimension, Optimizer optimizer, int... maxUniqueValuesPerFeature) {
        return inputAllEmbeddings(embeddingDimension, optimizer, maxUniqueValuesPerFeature, null);
    }
    
    /**
     * Multiple DIFFERENT embedding features with INDEPENDENT vocabularies and optional meaningful names.
     * 
     * <p>Same as {@link #inputAllEmbeddings(int, Optimizer, int...)} but with explicit feature names
     * for use with Map-based inputs in SimpleNet wrappers.
     * 
     * <p><b>Example - Ad Tech Features with names:</b>
     * <pre>{@code
     * // 3 different ID types with meaningful names
     * Layers.inputAllEmbeddings(64, optimizer,
     *     new int[]{100000, 50000, 25000},  // vocabulary sizes
     *     new String[]{"bundle_id", "publisher_id", "category_id"}  // names
     * )
     * // Now can use: model.train(Map.of("bundle_id", 12345, "publisher_id", 678, "category_id", 42), target)
     * }</pre>
     * 
     * @param embeddingDimension size of embedding vectors for all features
     * @param optimizer optimizer for embedding tables
     * @param maxUniqueValuesPerFeature vocabulary size for each feature (in input order)
     * @param featureNames optional array of meaningful names for each feature (can be null)
     * @return mixed feature input layer spec with all embedding features
     * @throws IllegalArgumentException if featureNames is non-null and length doesn't match maxUniqueValuesPerFeature length
     */
    public static Layer.Spec inputAllEmbeddings(int embeddingDimension, Optimizer optimizer, 
                                               int[] maxUniqueValuesPerFeature, String[] featureNames) {
        // Validate parameters
        if (embeddingDimension <= 0)
            throw new IllegalArgumentException("Embedding dimension must be positive: " + embeddingDimension);
        if (maxUniqueValuesPerFeature == null)
            throw new IllegalArgumentException("maxUniqueValuesPerFeature cannot be null");
        if (maxUniqueValuesPerFeature.length == 0)
            throw new IllegalArgumentException(
                "At least one feature must be configured. " +
                "Example: Layers.inputAllEmbeddings(64, optimizer, new int[]{1000, 500, 2000}, null)");
        if (featureNames != null && featureNames.length != maxUniqueValuesPerFeature.length)
            throw new IllegalArgumentException(String.format(
                "Feature names count (%d) must match vocabulary sizes count (%d)",
                featureNames.length, maxUniqueValuesPerFeature.length));
        
        Feature[] features = new Feature[maxUniqueValuesPerFeature.length];
        for (int i = 0; i < maxUniqueValuesPerFeature.length; i++) {
            if (maxUniqueValuesPerFeature[i] <= 0)
                throw new IllegalArgumentException(String.format(
                    "Feature %d%s: maxUniqueValues must be positive, got %d", 
                    i, featureNames != null ? " (" + featureNames[i] + ")" : "", maxUniqueValuesPerFeature[i]));
            
            if (featureNames != null) {
                features[i] = Feature.embedding(maxUniqueValuesPerFeature[i], embeddingDimension, featureNames[i]);
            } else {
                features[i] = Feature.embedding(maxUniqueValuesPerFeature[i], embeddingDimension);
            }
        }
        return MixedFeatureInputLayer.spec(optimizer, features, WeightInitStrategy.XAVIER);
    }
    
    /**
     * All features encoded as one-hot vectors.
     * 
     * <p>Use when all your features are low-cardinality categories that don't
     * need learned representations. Memory-efficient for small categorical features.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // All features are simple categories
     * Layers.inputAllOneHot(optimizer, 4, 8, 3, 7)
     * // connection_type (4), device_type (8), day_of_week (3), hour_bucket (7)
     * }</pre>
     * 
     * @param optimizer optimizer (not used since one-hot has no learnable parameters, but kept for API consistency)
     * @param numberOfCategoriesPerFeature category count for each feature (in input order)
     * @return mixed feature input layer spec with all one-hot features
     */
    public static Layer.Spec inputAllOneHot(Optimizer optimizer, int... numberOfCategoriesPerFeature) {
        return inputAllOneHot(optimizer, numberOfCategoriesPerFeature, null);
    }
    
    /**
     * All features encoded as one-hot vectors with optional meaningful names.
     * 
     * <p>Same as {@link #inputAllOneHot(Optimizer, int...)} but with explicit feature names
     * for use with Map-based inputs in SimpleNet wrappers.
     * 
     * <p><b>Example with names:</b>
     * <pre>{@code
     * // All features are simple categories with names
     * Layers.inputAllOneHot(optimizer, 
     *     new int[]{4, 8, 3, 7},
     *     new String[]{"connection_type", "device_type", "day_of_week", "hour_bucket"}
     * )
     * // Now can use: model.train(Map.of("connection_type", 2, "device_type", 5, ...), target)
     * }</pre>
     * 
     * @param optimizer optimizer (not used since one-hot has no learnable parameters, but kept for API consistency)
     * @param numberOfCategoriesPerFeature category count for each feature (in input order)
     * @param featureNames optional array of meaningful names for each feature (can be null)
     * @return mixed feature input layer spec with all one-hot features
     * @throws IllegalArgumentException if featureNames is non-null and length doesn't match numberOfCategoriesPerFeature length
     */
    public static Layer.Spec inputAllOneHot(Optimizer optimizer, int[] numberOfCategoriesPerFeature, String[] featureNames) {
        // Validate parameters to prevent configuration errors
        if (numberOfCategoriesPerFeature == null)
            throw new IllegalArgumentException("numberOfCategoriesPerFeature cannot be null");
        if (numberOfCategoriesPerFeature.length == 0)
            throw new IllegalArgumentException(
                "At least one feature must be configured. " +
                "Example: Layers.inputAllOneHot(optimizer, new int[]{4, 8, 3}, null)");
        if (featureNames != null && featureNames.length != numberOfCategoriesPerFeature.length)
            throw new IllegalArgumentException(String.format(
                "Feature names count (%d) must match categories count (%d)",
                featureNames.length, numberOfCategoriesPerFeature.length));
        
        Feature[] features = new Feature[numberOfCategoriesPerFeature.length];
        for (int i = 0; i < numberOfCategoriesPerFeature.length; i++) {
            if (numberOfCategoriesPerFeature[i] <= 0)
                throw new IllegalArgumentException(String.format(
                    "Feature %d%s: numberOfCategories must be positive, got %d", 
                    i, featureNames != null ? " (" + featureNames[i] + ")" : "", numberOfCategoriesPerFeature[i]));
            
            if (featureNames != null) {
                features[i] = Feature.oneHot(numberOfCategoriesPerFeature[i], featureNames[i]);
            } else {
                features[i] = Feature.oneHot(numberOfCategoriesPerFeature[i]);
            }
        }
        return MixedFeatureInputLayer.spec(optimizer, features, WeightInitStrategy.XAVIER);
    }
    
    /**
     * All features encoded as one-hot vectors with shared vocabulary size.
     * 
     * <p>Use for language models where all positions in a sequence share the same vocabulary.
     * This is a convenience method that avoids having to specify the vocabulary size
     * multiple times for each position.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // Language model with 20-token sequences, 5000-word vocabulary
     * NeuralNet.newBuilder()
     *     .input(20)  // 20 tokens
     *     .layer(Layers.inputAllOneHotShared(20, 5000, optimizer))
     *     .layer(Layers.hiddenDenseRelu(128))
     *     .output(Layers.outputSoftmaxCrossEntropy(5000))
     * 
     * // Equivalent to: Layers.inputAllOneHot(optimizer, 5000, 5000, 5000, ... [20 times])
     * }</pre>
     * 
     * @param sequenceLength number of positions in the sequence
     * @param sharedVocabSize vocabulary size shared by all positions
     * @param optimizer optimizer (not used for one-hot, but kept for consistency)
     * @return mixed feature input layer spec with shared one-hot features
     */
    public static Layer.Spec inputAllOneHotShared(int sequenceLength, int sharedVocabSize, 
                                                  Optimizer optimizer) {
        // Validate parameters
        if (sequenceLength <= 0)
            throw new IllegalArgumentException("sequenceLength must be positive, got " + sequenceLength);
        if (sharedVocabSize <= 0)
            throw new IllegalArgumentException("sharedVocabSize must be positive, got " + sharedVocabSize);
        
        // Create features array with same vocab size for each position
        Feature[] features = new Feature[sequenceLength];
        for (int i = 0; i < sequenceLength; i++) {
            features[i] = Feature.oneHot(sharedVocabSize);
        }
        return MixedFeatureInputLayer.spec(optimizer, features, WeightInitStrategy.XAVIER);
    }
    
    /**
     * All features encoded as one-hot vectors with shared vocabulary size (default optimizer).
     * 
     * <p>Uses the neural network's default optimizer set with {@code .setDefaultOptimizer()}.
     * Convenient for language models where all layers use the same optimizer.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * NeuralNet.newBuilder()
     *     .setDefaultOptimizer(new AdamWOptimizer(0.001f))
     *     .input(20)  // 20 tokens
     *     .layer(Layers.inputAllOneHotShared(20, 5000))  // Uses default optimizer
     *     .layer(Layers.hiddenDenseRelu(128))
     *     .output(Layers.outputSoftmaxCrossEntropy(5000))
     * }</pre>
     * 
     * @param sequenceLength number of positions in the sequence
     * @param sharedVocabSize vocabulary size shared by all positions
     * @return mixed feature input layer spec using default optimizer
     */
    public static Layer.Spec inputAllOneHotShared(int sequenceLength, int sharedVocabSize) {
        return inputAllOneHotShared(sequenceLength, sharedVocabSize, null);
    }
    
    /**
     * All features passed through as numerical values.
     * 
     * <p>Use for pure numerical/continuous feature sets where no encoding is needed.
     * Input values are copied directly to output without transformation.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // All features are continuous values
     * Layers.inputAllNumerical(5)
     * // age, price, duration, rating, distance → 5 pass-through values
     * }</pre>
     * 
     * @param numberOfFeatures number of numerical features in input
     * @return mixed feature input layer spec with all pass-through features
     */
    public static Layer.Spec inputAllNumerical(int numberOfFeatures) {
        return inputAllNumerical(numberOfFeatures, null);
    }
    
    /**
     * All features passed through as numerical values with optional meaningful names.
     * 
     * <p>Same as {@link #inputAllNumerical(int)} but with explicit feature names
     * for use with Map-based inputs in SimpleNet wrappers.
     * 
     * <p><b>Example with names:</b>
     * <pre>{@code
     * // 5 numerical features with meaningful names
     * Layers.inputAllNumerical(5, 
     *     new String[]{"temperature", "humidity", "wind_speed", "pressure", "rainfall"})
     * // Now can use: model.train(Map.of("temperature", 23.5f, "humidity", 65.0f, ...), target)
     * }</pre>
     * 
     * @param numberOfFeatures number of numerical features in input
     * @param featureNames optional array of meaningful names for each feature (can be null)
     * @return mixed feature input layer spec with all pass-through features
     * @throws IllegalArgumentException if featureNames is non-null and length doesn't match numberOfFeatures
     */
    public static Layer.Spec inputAllNumerical(int numberOfFeatures, String[] featureNames) {
        // Validate parameters to prevent configuration errors
        if (numberOfFeatures <= 0)
            throw new IllegalArgumentException(
                "Number of features must be positive: " + numberOfFeatures + ". " +
                "Example: Layers.inputAllNumerical(5) for 5 numerical features");
        if (featureNames != null && featureNames.length != numberOfFeatures)
            throw new IllegalArgumentException(String.format(
                "Feature names count (%d) must match number of features (%d)",
                featureNames.length, numberOfFeatures));
        
        Feature[] features = new Feature[numberOfFeatures];
        for (int i = 0; i < numberOfFeatures; i++) {
            if (featureNames != null) {
                features[i] = Feature.passthrough(featureNames[i]);
            } else {
                features[i] = Feature.passthrough();
            }
        }
        return MixedFeatureInputLayer.spec(null, features, WeightInitStrategy.XAVIER);
    }

    // ===============================
    // RECURRENT LAYERS
    // ===============================
    
    /**
     * @deprecated Use hiddenGruAll() or hiddenGruLast() for clarity about output shape
     */
    @Deprecated
    public static Layer.Spec gru(int hiddenSize, Optimizer optimizer) {
        return GruLayer.spec(hiddenSize, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * @deprecated Use hiddenGruAll() or hiddenGruLast() for clarity about output shape
     */
    @Deprecated
    public static Layer.Spec gru(int hiddenSize) {
        return GruLayer.spec(hiddenSize, null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * @deprecated Use hiddenGruAll() or hiddenGruLast() for clarity about output shape
     */
    @Deprecated
    public static Layer.Spec gru(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return GruLayer.spec(hiddenSize, optimizer, initStrategy);
    }
    
    /**
     * GRU layer that outputs ALL timesteps - for sequence-to-sequence models.
     * 
     * <p><b>Output shape:</b> [sequenceLength × hiddenSize]
     * 
     * <p><b>When to use:</b>
     * <ul>
     *   <li><b>Sequence-to-sequence:</b> Machine translation, text summarization</li>
     *   <li><b>Attention mechanisms:</b> When you need all hidden states for attention</li>
     *   <li><b>Bidirectional RNNs:</b> Processing sequences in both directions</li>
     *   <li><b>Time series analysis:</b> When every timestep matters</li>
     * </ul>
     * 
     * <p><b>Example - Machine Translation:</b>
     * <pre>{@code
     * NeuralNet encoder = NeuralNet.newBuilder()
     *     .input(50)  // source sequence length
     *     .layer(Layers.inputEmbedding(30000, 256))
     *     .layer(Layers.hiddenGruAll(512))  // Outputs: 50 × 512
     *     .layer(Layers.hiddenGruAll(512))  // Stack multiple GRUs
     *     .output(...);  // Use all states for attention
     * }</pre>
     * 
     * @param hiddenSize number of hidden units in GRU cell
     * @param optimizer optimizer for GRU parameters
     * @return GRU spec that outputs all timesteps
     */
    public static Layer.Spec hiddenGruAll(int hiddenSize, Optimizer optimizer) {
        return GruLayer.specAll(hiddenSize, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * GRU layer that outputs ALL timesteps - using default optimizer.
     * 
     * @param hiddenSize number of hidden units
     * @return GRU spec that outputs all timesteps
     */
    public static Layer.Spec hiddenGruAll(int hiddenSize) {
        return GruLayer.specAll(hiddenSize, null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * GRU layer that outputs ALL timesteps - with custom initialization.
     * 
     * @param hiddenSize number of hidden units
     * @param optimizer optimizer for parameters
     * @param initStrategy weight initialization strategy
     * @return GRU spec that outputs all timesteps
     */
    public static Layer.Spec hiddenGruAll(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return GruLayer.specAll(hiddenSize, optimizer, initStrategy);
    }
    
    /**
     * GRU layer that outputs ALL timesteps - with input dimension hint.
     * 
     * <p>Use this version when you know the per-timestep input size (e.g., embedding dimension).
     * This enables proper output size calculation before layer creation.
     * 
     * <p><b>Example - After embedding layer:</b>
     * <pre>{@code
     * int embeddingDim = 256;
     * NeuralNet model = NeuralNet.newBuilder()
     *     .input(50)  // sequence length
     *     .layer(Layers.inputEmbedding(30000, embeddingDim))
     *     .layer(Layers.hiddenGruAll(512, optimizer, embeddingDim))  // Knows input is 256 per timestep
     *     .output(...);
     * }</pre>
     * 
     * @param hiddenSize number of hidden units
     * @param optimizer optimizer for parameters
     * @param expectedInputDimension expected size per timestep (e.g., embedding dimension)
     * @return GRU spec that outputs all timesteps with proper size calculation
     */
    public static Layer.Spec hiddenGruAll(int hiddenSize, Optimizer optimizer, int expectedInputDimension) {
        return GruLayer.specAll(hiddenSize, optimizer, WeightInitStrategy.XAVIER, expectedInputDimension);
    }
    
    /**
     * GRU layer that outputs ONLY LAST timestep - for classification/prediction.
     * 
     * <p><b>Output shape:</b> [hiddenSize]
     * 
     * <p><b>When to use:</b>
     * <ul>
     *   <li><b>Sequence classification:</b> Sentiment analysis, spam detection</li>
     *   <li><b>Language modeling:</b> Predicting next word/character</li>
     *   <li><b>Time series prediction:</b> Forecasting next value</li>
     *   <li><b>Any many-to-one task:</b> Whole sequence → single output</li>
     * </ul>
     * 
     * <p><b>Example - Sentiment Classification:</b>
     * <pre>{@code
     * NeuralNet classifier = NeuralNet.newBuilder()
     *     .input(100)  // max sequence length
     *     .layer(Layers.inputEmbedding(10000, 128))
     *     .layer(Layers.hiddenGruLast(256))  // Outputs: 256 (just final state)
     *     .layer(Layers.hiddenDenseRelu(128))
     *     .output(Layers.outputSoftmaxCrossEntropy(2));  // positive/negative
     * }</pre>
     * 
     * @param hiddenSize number of hidden units in GRU cell
     * @param optimizer optimizer for GRU parameters
     * @return GRU spec that outputs only last timestep
     */
    public static Layer.Spec hiddenGruLast(int hiddenSize, Optimizer optimizer) {
        return GruLayer.specLast(hiddenSize, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * GRU layer that outputs ONLY LAST timestep - using default optimizer.
     * 
     * @param hiddenSize number of hidden units
     * @return GRU spec that outputs only last timestep
     */
    public static Layer.Spec hiddenGruLast(int hiddenSize) {
        return GruLayer.specLast(hiddenSize, null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * GRU layer that outputs ONLY LAST timestep - with custom initialization.
     * 
     * @param hiddenSize number of hidden units
     * @param optimizer optimizer for parameters
     * @param initStrategy weight initialization strategy
     * @return GRU spec that outputs only last timestep
     */
    public static Layer.Spec hiddenGruLast(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return GruLayer.specLast(hiddenSize, optimizer, initStrategy);
    }
    
    // ===============================
    // LAYER-NORMALIZED GRU VARIANTS
    // ===============================
    
    /**
     * Layer-normalized GRU that outputs ALL timesteps.
     * 
     * <p><b>What is Layer Normalization?</b> Normalizes activations to stabilize training,
     * allowing higher learning rates and improving convergence. Especially beneficial for RNNs.
     * 
     * <p><b>When to use:</b>
     * <ul>
     *   <li>Small datasets (like PTB) where training stability is crucial</li>
     *   <li>When you want faster convergence with higher learning rates</li>
     *   <li>Language models and sequence tasks</li>
     * </ul>
     * 
     * @param hiddenSize number of hidden units
     * @param optimizer optimizer for parameters
     * @return Layer-normalized GRU spec that outputs all timesteps
     */
    public static Layer.Spec hiddenGruAllNormalized(int hiddenSize, Optimizer optimizer) {
        return GruLayer.specAllNormalized(hiddenSize, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Layer-normalized GRU that outputs ALL timesteps - using default optimizer.
     * 
     * @param hiddenSize number of hidden units
     * @return Layer-normalized GRU spec that outputs all timesteps
     */
    public static Layer.Spec hiddenGruAllNormalized(int hiddenSize) {
        return GruLayer.specAllNormalized(hiddenSize, null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Layer-normalized GRU that outputs ONLY LAST timestep.
     * 
     * <p>Perfect for classification tasks where you only need the final hidden state.
     * The layer normalization helps stabilize training, especially on small datasets.
     * 
     * @param hiddenSize number of hidden units
     * @param optimizer optimizer for parameters
     * @return Layer-normalized GRU spec that outputs only last timestep
     */
    public static Layer.Spec hiddenGruLastNormalized(int hiddenSize, Optimizer optimizer) {
        return GruLayer.specLastNormalized(hiddenSize, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Layer-normalized GRU that outputs ONLY LAST timestep - using default optimizer.
     * 
     * @param hiddenSize number of hidden units
     * @return Layer-normalized GRU spec that outputs only last timestep
     */
    public static Layer.Spec hiddenGruLastNormalized(int hiddenSize) {
        return GruLayer.specLastNormalized(hiddenSize, null, WeightInitStrategy.XAVIER);
    }

    // ===============================
    // HIDDEN LAYERS (No Loss Computation)
    // ===============================
    
    /**
     * Dense (fully-connected) hidden layer.
     * Most common layer type for traditional neural networks.
     */
    public static Layer.Spec hiddenDense(int neurons, Activator activator, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return DenseLayer.spec(neurons, activator, optimizer, initStrategy);
    }
    
    /**
     * Dense layer with custom learning rate ratio.
     * Allows fine-tuning the learning speed of specific layers.
     * 
     * @param neurons number of neurons
     * @param activator activation function
     * @param optimizer optimizer (null to use default)
     * @param initStrategy weight initialization strategy
     * @param learningRateRatio learning rate scaling (1.0 = normal, 0.1 = 10x slower)
     */
    public static Layer.Spec hiddenDense(int neurons, Activator activator, Optimizer optimizer, 
                                         WeightInitStrategy initStrategy, double learningRateRatio) {
        return DenseLayer.spec(neurons, activator, optimizer, initStrategy, learningRateRatio);
    }
    
    /**
     * Dense layer with automatic weight initialization based on activator.
     */
    public static Layer.Spec hiddenDense(int neurons, Activator activator, Optimizer optimizer) {
        WeightInitStrategy strategy = activator instanceof ReluActivator ? WeightInitStrategy.HE : WeightInitStrategy.XAVIER;
        return hiddenDense(neurons, activator, optimizer, strategy);
    }
    
    /**
     * ReLU dense layer with chainable configuration (uses default optimizer).
     * Most common layer type - allows setting optimizer and learning rate ratio via chaining.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * .layer(Layers.hiddenDenseRelu(256))                           // Uses default optimizer
     * .layer(Layers.hiddenDenseRelu(128).learningRateRatio(0.5f))  // 2x slower learning
     * .layer(Layers.hiddenDenseRelu(64).optimizer(new SGD(0.01f))) // Custom optimizer
     * }</pre>
     */
    public static DenseLayer.DenseLayerSpec hiddenDenseRelu(int neurons) {
        return DenseLayer.specChainable(neurons, ReluActivator.INSTANCE, WeightInitStrategy.HE);
    }
    
    /**
     * ReLU dense layer (most common combination).
     */
    public static Layer.Spec hiddenDenseRelu(int neurons, Optimizer optimizer) {
        return hiddenDense(neurons, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE);
    }
    
    
    /**
     * Leaky ReLU dense layer with chainable configuration (uses default optimizer).
     * Helps avoid "dying ReLU" problems by allowing small gradients for negative inputs.
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * .layer(Layers.hiddenDenseLeakyRelu(256))                           // Default alpha=0.01
     * .layer(Layers.hiddenDenseLeakyRelu(128).learningRateRatio(0.5f))  // With LR adjustment
     * .layer(Layers.hiddenDenseLeakyRelu(64).optimizer(new SGD(0.01f))) // Custom optimizer
     * }</pre>
     */
    public static DenseLayer.DenseLayerSpec hiddenDenseLeakyRelu(int neurons) {
        return DenseLayer.specChainable(neurons, LeakyReluActivator.createDefault(), WeightInitStrategy.HE_PLUS_UNIFORM_NOISE);
    }
    
    /**
     * Leaky ReLU dense layer with default alpha=0.01.
     */
    public static Layer.Spec hiddenDenseLeakyRelu(int neurons, Optimizer optimizer) {
        return hiddenDense(neurons, LeakyReluActivator.createDefault(), optimizer, WeightInitStrategy.HE_PLUS_UNIFORM_NOISE);
    }
    
    /**
     * Leaky ReLU dense layer with custom alpha value.
     * 
     * @param neurons number of neurons
     * @param alpha negative slope (typically 0.01 to 0.3)
     */
    public static DenseLayer.DenseLayerSpec hiddenDenseLeakyRelu(int neurons, float alpha) {
        return DenseLayer.specChainable(neurons, LeakyReluActivator.create(alpha), WeightInitStrategy.HE_PLUS_UNIFORM_NOISE);
    }
    
    /**
     * Leaky ReLU dense layer with custom alpha and optimizer.
     */
    public static Layer.Spec hiddenDenseLeakyRelu(int neurons, float alpha, Optimizer optimizer) {
        return hiddenDense(neurons, LeakyReluActivator.create(alpha), optimizer, WeightInitStrategy.HE_PLUS_UNIFORM_NOISE);
    }
    
    /**
     * ReLU dense layer with custom learning rate ratio.
     * Useful for transfer learning or fine-tuning specific layers.
     * 
     * @param neurons number of neurons
     * @param learningRateRatio learning rate scaling (0.1 = 10x slower, 2.0 = 2x faster)
     */
    public static Layer.Spec hiddenDenseRelu(int neurons, double learningRateRatio) {
        return hiddenDense(neurons, ReluActivator.INSTANCE, null, WeightInitStrategy.HE, learningRateRatio);
    }
    
    /**
     * Tanh dense layer.
     */
    public static Layer.Spec hiddenDenseTanh(int neurons, Optimizer optimizer) {
        return hiddenDense(neurons, TanhActivator.INSTANCE, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Tanh dense layer using default optimizer.
     */
    public static Layer.Spec hiddenDenseTanh(int neurons) {
        return hiddenDense(neurons, TanhActivator.INSTANCE, null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Sigmoid dense layer.
     */
    public static Layer.Spec hiddenDenseSigmoid(int neurons, Optimizer optimizer) {
        return hiddenDense(neurons, SigmoidActivator.INSTANCE, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Sigmoid dense layer using default optimizer.
     */
    public static Layer.Spec hiddenDenseSigmoid(int neurons) {
        return hiddenDense(neurons, SigmoidActivator.INSTANCE, null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Linear dense layer (no activation).
     * Useful for residual connections, pre-output layers, and skip connections.
     */
    public static Layer.Spec hiddenDenseLinear(int neurons, Optimizer optimizer) {
        return hiddenDense(neurons, LinearActivator.INSTANCE, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Linear dense layer using default optimizer.
     */
    public static Layer.Spec hiddenDenseLinear(int neurons) {
        return hiddenDense(neurons, LinearActivator.INSTANCE, null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Dropout layer for regularization.
     * 
     * <p><b>What it does:</b> Randomly drops connections during training to prevent overfitting.
     * During inference, all connections are active (standard inverted dropout).
     * 
     * <p><b>Placement guidelines:</b>
     * <ul>
     *   <li>After activation functions (ReLU, tanh, etc.)</li>
     *   <li>NOT after batch normalization</li>
     *   <li>NOT in the input layer</li>
     *   <li>Often NOT in the output layer</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * NeuralNet model = NeuralNet.newBuilder()
     *     .input(784)
     *     .layer(Layers.hiddenDenseRelu(256))
     *     .layer(Layers.dropout(0.5))           // 50% dropout after first hidden layer
     *     .layer(Layers.hiddenDenseRelu(128))
     *     .layer(Layers.dropout(0.3))           // 30% dropout after second hidden layer
     *     .output(Layers.outputSoftmaxCrossEntropy(10));
     * }</pre>
     * 
     * @param dropoutRate probability of dropping each neuron (0.0 to 1.0)
     * @return dropout layer spec
     */
    public static Layer.Spec dropout(float dropoutRate) {
        return DropoutLayer.spec(dropoutRate);
    }
    
    /**
     * Standard dropout with 0.5 rate (original paper recommendation).
     * 
     * <p>50% dropout is the most common choice and was recommended in the original
     * dropout paper for hidden layers.
     * 
     * @return dropout layer spec with 0.5 dropout rate
     */
    public static Layer.Spec dropout() {
        return dropout(0.5f);
    }
    
    /**
     * Layer normalization for stabilizing training.
     * 
     * <p><b>What it does:</b> Normalizes activations across features to zero mean
     * and unit variance, then applies learnable scale and shift.
     * 
     * <p><b>Benefits:</b>
     * <ul>
     *   <li>Stabilizes training, especially for RNNs</li>
     *   <li>Allows higher learning rates</li>
     *   <li>Acts as regularization</li>
     *   <li>Works with any batch size</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * NeuralNet model = NeuralNet.newBuilder()
     *     .input(seqLength)
     *     .layer(Layers.inputSequenceEmbedding(seqLen, vocab, embedDim))
     *     .layer(Layers.hiddenGruAll(hiddenSize))
     *     .layer(Layers.layerNorm())  // Normalize GRU outputs
     *     .layer(Layers.hiddenDenseRelu(hiddenSize))
     *     .output(Layers.outputSoftmaxCrossEntropy(vocab));
     * }</pre>
     * 
     * @param optimizer optimizer for gamma and beta parameters
     * @return layer normalization spec
     */
    public static Layer.Spec layerNorm(Optimizer optimizer) {
        return LayerNormLayer.spec(optimizer);
    }
    
    /**
     * Layer normalization using default optimizer.
     * 
     * @return layer normalization spec with default optimizer
     */
    public static Layer.Spec layerNorm() {
        return LayerNormLayer.spec();
    }
    
    // Future hidden layer types can be added here:
    // public static Layer.Spec hiddenConv2D(...)
    // public static Layer.Spec hiddenLSTM(...)
    // public static Layer.Spec hiddenGRU(...)
    // public static Layer.Spec hiddenAttention(...)
    
    // ===============================
    // OUTPUT LAYERS (Handle Loss Computation)
    // ===============================
    
    /**
     * Softmax + Cross-Entropy output for multi-class classification.
     * 
     * This is the standard choice for:
     * - Image classification (MNIST, CIFAR, ImageNet)
     * - Text classification 
     * - Any task with mutually exclusive classes
     * 
     * Automatically handles numerically stable computation.
     */
    public static Layer.Spec outputSoftmaxCrossEntropy(int numClasses, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return SoftmaxCrossEntropyOutput.spec(numClasses, optimizer, initStrategy);
    }
    
    /**
     * Softmax + Cross-Entropy with Xavier initialization (recommended default).
     */
    public static Layer.Spec outputSoftmaxCrossEntropy(int numClasses, Optimizer optimizer) {
        return outputSoftmaxCrossEntropy(numClasses, optimizer, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Softmax + Cross-Entropy using default optimizer.
     * Simplest way to add a classification output layer.
     * 
     * Example:
     * <pre>{@code
     * NeuralNet net = NeuralNet.newBuilder()
     *     .input(784)
     *     .defaultOptimizer(new SgdOptimizer(0.01f))
     *     .layer(Layers.hiddenDenseRelu(256))  
     *     .output(Layers.outputSoftmaxCrossEntropy(10));  // 10 classes
     * }</pre>
     */
    public static Layer.Spec outputSoftmaxCrossEntropy(int numClasses) {
        return outputSoftmaxCrossEntropy(numClasses, null, WeightInitStrategy.XAVIER);
    }
    
    /**
     * Linear output for regression tasks.
     * No activation, typically used with MSE loss.
     */
    public static Layer.Spec outputLinearRegression(int outputs, Optimizer optimizer) {
        return LinearRegressionOutput.spec(outputs, optimizer);
    }
    
    /**
     * Linear output using default optimizer.
     * 
     * @param outputs number of regression outputs
     * @return linear regression output spec
     */
    public static Layer.Spec outputLinearRegression(int outputs) {
        return LinearRegressionOutput.spec(outputs, null);
    }
    
    /**
     * Huber regression output - robust alternative to MSE.
     * 
     * <p>Less sensitive to outliers than standard MSE loss.
     * Combines quadratic loss for small errors with linear loss for large errors.
     * 
     * <p>Use for:
     * <ul>
     *   <li>Regression with potential outliers
     *   <li>Robust parameter estimation
     *   <li>Any task where extreme errors should have limited influence
     * </ul>
     * 
     * @param outputs number of output values
     * @param optimizer optimizer for this layer
     * @return Huber regression output spec
     */
    public static Layer.Spec outputHuberRegression(int outputs, Optimizer optimizer) {
        return HuberRegressionOutput.spec(outputs, optimizer);
    }
    
    /**
     * Huber regression output with custom delta threshold.
     * 
     * <p>Delta controls the transition from quadratic to linear loss:
     * <ul>
     *   <li>Small delta (0.5): More MSE-like, sensitive to small errors
     *   <li>Default delta (1.0): Balanced behavior
     *   <li>Large delta (2.0+): More MAE-like, robust to outliers
     * </ul>
     * 
     * @param outputs number of output values
     * @param optimizer optimizer for this layer
     * @param delta threshold parameter (typically 0.5 to 2.0)
     * @return Huber regression output spec
     */
    public static Layer.Spec outputHuberRegression(int outputs, Optimizer optimizer, float delta) {
        return HuberRegressionOutput.spec(outputs, optimizer, delta);
    }
    
    /**
     * Huber regression output using default optimizer and delta=1.0.
     * 
     * @param outputs number of output values
     * @return Huber regression output spec
     */
    public static Layer.Spec outputHuberRegression(int outputs) {
        return HuberRegressionOutput.spec(outputs, null);
    }
    
    /**
     * Sigmoid + Binary Cross-Entropy for binary classification.
     */
    public static Layer.Spec outputSigmoidBinary(Optimizer optimizer) {
        return SigmoidBinaryCrossEntropyOutput.spec(optimizer);
    }
    
    /**
     * Sigmoid + Binary Cross-Entropy using default optimizer.
     * 
     * @return binary classification output spec
     */
    public static Layer.Spec outputSigmoidBinary() {
        return SigmoidBinaryCrossEntropyOutput.spec(null);
    }
    
    /**
     * Multi-label sigmoid output (multiple independent binary classifications).
     */
    public static Layer.Spec
    outputMultiLabel(int numLabels, Optimizer optimizer) {
        return MultiLabelSigmoidOutput.spec(numLabels, optimizer);
    }
    
    /**
     * Multi-label sigmoid output using default optimizer.
     * 
     * @param numLabels number of independent labels
     * @return multi-label output spec
     */
    public static Layer.Spec outputMultiLabel(int numLabels) {
        return MultiLabelSigmoidOutput.spec(numLabels, null);
    }
    
    // Future output layer types:
    // public static Layer.Spec outputContrastiveLoss(...)
    // public static Layer.Spec outputTripletLoss(...)
    // public static Layer.Spec outputCTC(...) // For speech/text
}