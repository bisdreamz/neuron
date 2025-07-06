package dev.neuronic.net.layers;

import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.AdamOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationRegistry;
import dev.neuronic.net.Dictionary;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;

/**
 * Input layer for SEQUENCE modeling where all positions share the SAME vocabulary.
 * 
 * <p><b>CRITICAL DISTINCTION from InputAllEmbeddings:</b>
 * <ul>
 *   <li><b>InputAllEmbeddings:</b> Multiple DIFFERENT vocabularies (bundle_id vocab, publisher_id vocab, etc.)</li>
 *   <li><b>InputSequenceEmbedding:</b> One SHARED vocabulary across all sequence positions</li>
 * </ul>
 * 
 * <p><b>What it does:</b> Converts sequences of words/tokens into dense vector representations
 * where each position in the sequence looks up embeddings from the SAME vocabulary table.
 * Automatically builds vocabulary from training data like SimpleNet.
 * 
 * <p><b>Perfect for:</b>
 * <ul>
 *   <li><b>Language models (GPT, BERT):</b> Words in sequence share vocabulary</li>
 *   <li><b>Time series with categories:</b> Same categories appear across time</li>
 *   <li><b>DNA/protein sequences:</b> Same alphabet (ACGT) across positions</li>
 *   <li><b>Music sequences:</b> Same note vocabulary across time</li>
 * </ul>
 * 
 * <p><b>How it works:</b>
 * <ol>
 *   <li>Maintains ONE embedding table for the entire vocabulary</li>
 *   <li>Each position in sequence looks up its word in the SAME table</li>
 *   <li>Automatically builds vocabulary from training data</li>
 *   <li>Outputs concatenated embeddings: [seqLen × embeddingDim]</li>
 * </ol>
 * 
 * <p><b>Example Usage - Language Model:</b>
 * <pre>{@code
 * // WikiText-2 language model with 35-word context window
 * NeuralNet model = NeuralNet.newBuilder()
 *     .input(35)  // sequence length
 *     .layer(Layers.inputSequenceEmbedding(35, 30000, 256, optimizer))
 *     .layer(Layers.gru(512))
 *     .output(Layers.outputSoftmaxCrossEntropy(30000));
 * 
 * SimpleNetString lm = SimpleNet.ofStringClassification(model);
 * 
 * // Train with string sequences - vocabulary builds automatically!
 * String[] context = {"The", "quick", "brown", "fox", "jumps", ...};  // 35 words
 * String nextWord = "over";
 * lm.train(context, nextWord);
 * 
 * // All 35 positions share the same 30k vocabulary
 * // Total embeddings: 30k × 256 (not 35 × 30k × 256!)
 * }</pre>
 * 
 * <p><b>Memory Usage:</b> vocabSize × embeddingDim × 4 bytes
 * <br>Example: 30k vocab × 256 dims = ~30MB (shared across all positions)
 * 
 * <p><b>Thread Safety:</b> Fully thread-safe for concurrent training and inference.
 */
public class InputSequenceEmbeddingLayer implements Layer, Serializable {
    
    private static final String UNK_TOKEN = "<unk>";
    private static final int UNK_INDEX = 0;
    
    private final Optimizer optimizer;
    private Optimizer embeddingOptimizer; // Optimizer for embeddings (may be same as optimizer)
    private final Dictionary vocabulary;
    private final float[][] embeddings; // [vocabSize][embeddingDim]
    private final int sequenceLength;
    private final int maxVocabSize;
    private final int embeddingDim;
    private final ThreadLocal<float[]> outputBuffers;
    private final ThreadLocal<float[][]> embeddingGradientBuffers;
    private final float[] emptyBiases = new float[0];  // Reuse same instance
    private final float[] emptyBiasGradients = new float[0];  // Reuse same instance
    
    public InputSequenceEmbeddingLayer(Optimizer optimizer, int sequenceLength, int maxVocabSize, 
                                      int embeddingDim, WeightInitStrategy initStrategy, FastRandom random) {
        if (sequenceLength <= 0)
            throw new IllegalArgumentException("Sequence length must be positive: " + sequenceLength);
        if (maxVocabSize <= 0)
            throw new IllegalArgumentException("Max vocab size must be positive: " + maxVocabSize);
        if (embeddingDim <= 0)
            throw new IllegalArgumentException("Embedding dim must be positive: " + embeddingDim);
        
        this.optimizer = optimizer;
        this.embeddingOptimizer = optimizer.forEmbeddings();
        this.sequenceLength = sequenceLength;
        this.maxVocabSize = maxVocabSize;
        this.embeddingDim = embeddingDim;
        this.vocabulary = new Dictionary(maxVocabSize);
        this.embeddings = new float[maxVocabSize][embeddingDim];
        this.outputBuffers = ThreadLocal.withInitial(() -> new float[sequenceLength * embeddingDim]);
        this.embeddingGradientBuffers = ThreadLocal.withInitial(() -> new float[maxVocabSize][embeddingDim]);
        
        // Reserve index 0 for <unk> token
        // Dictionary will auto-assign indices, so we need to ensure <unk> gets 0
        vocabulary.getIndex(UNK_TOKEN); // This will assign index 0
        
        // Initialize embeddings with uniform distribution for better learning
        // Embeddings need different initialization than dense layers
        NetMath.embeddingInitUniform(embeddings, -0.05f, 0.05f, random);
    }
    
    @Override
    public LayerContext forward(float[] input, boolean isTraining) {
        // Support dual mode: float arrays containing token IDs for bulk training
        if (input.length != sequenceLength) {
            throw new IllegalArgumentException(String.format(
                "Expected sequence of %d tokens, got %d",
                sequenceLength, input.length));
        }
        
        float[] output = outputBuffers.get();
        
        // Process token IDs directly
        for (int i = 0; i < input.length; i++) {
            int tokenId = (int) input[i];
            
            // Validate token ID
            if (tokenId != input[i]) {
                throw new IllegalArgumentException(
                    "Input must contain integer token IDs, got: " + input[i]);
            }
            if (tokenId < 0 || tokenId >= maxVocabSize) {
                throw new IllegalArgumentException(
                    "Invalid token ID: " + tokenId + " (vocab size: " + maxVocabSize + ")");
            }
            
            // Look up embedding
            System.arraycopy(embeddings[tokenId], 0, output, i * embeddingDim, embeddingDim);
        }
        
        // Create result array with exactly the right size and use defensive copying
        float[] result = new float[sequenceLength * embeddingDim];
        System.arraycopy(output, 0, result, 0, result.length);
        return new LayerContext(input.clone(), null, result);
    }
    
    @Override
    public LayerContext forward(float[] input, ExecutorService executor) {
        return forward(input, false);
    }
    
    /**
     * Special forward method for string sequences.
     * This is called by SimpleNet when it detects an InputSequenceEmbeddingLayer.
     */
    public LayerContext forwardSequence(Object input) {
        String[] words = convertToStringArray(input);
        
        if (words.length != sequenceLength) {
            throw new IllegalArgumentException(String.format(
                "Expected sequence of %d words, got %d. " +
                "Sequences must be exactly %d words for this model.",
                sequenceLength, words.length, sequenceLength));
        }
        
        float[] output = outputBuffers.get();
        
        // Convert each word to embedding using SHARED vocabulary
        for (int i = 0; i < words.length; i++) {
            int tokenId = getOrAssignTokenId(words[i]);
            System.arraycopy(embeddings[tokenId], 0, output, i * embeddingDim, embeddingDim);
        }
        
        // Create dummy float array for LayerContext compatibility  
        float[] dummyInput = new float[0];
        // Use safe factory method approach - clone output buffer for SequenceLayerContext
        float[] result = new float[sequenceLength * embeddingDim];
        System.arraycopy(output, 0, result, 0, result.length);
        return new SequenceLayerContext(dummyInput, null, result, words);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient, ExecutorService executor) {
        LayerContext context = stack[stackIndex];
        
        if (upstreamGradient.length != sequenceLength * embeddingDim)
            throw new IllegalArgumentException("Gradient length mismatch: expected " + 
                (sequenceLength * embeddingDim) + ", got " + upstreamGradient.length);
        
        float[][] embeddingGradients = embeddingGradientBuffers.get();
        
        // Clear gradients
        for (int i = 0; i < maxVocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                embeddingGradients[i][j] = 0.0f;
            }
        }
        
        // Handle both modes: string sequences and token IDs
        if (context instanceof SequenceLayerContext) {
            // String mode - backward compatibility
            SequenceLayerContext seqContext = (SequenceLayerContext) context;
            String[] words = seqContext.words;
            
            for (int i = 0; i < words.length; i++) {
                int tokenId;
                try {
                    tokenId = vocabulary.getIndex(words[i]);
                } catch (IllegalStateException e) {
                    // Dictionary is full - use UNK token
                    tokenId = UNK_INDEX;
                }
                if (tokenId == -1) {
                    tokenId = UNK_INDEX; // Word not in vocabulary during backward pass
                }
                
                int startIdx = i * embeddingDim;
                for (int j = 0; j < embeddingDim; j++) {
                    embeddingGradients[tokenId][j] += upstreamGradient[startIdx + j];
                }
            }
        } else {
            // Token ID mode - for bulk training
            float[] tokenIds = context.inputs();
            
            for (int i = 0; i < tokenIds.length; i++) {
                int tokenId = (int) tokenIds[i];
                int startIdx = i * embeddingDim;
                
                for (int j = 0; j < embeddingDim; j++) {
                    embeddingGradients[tokenId][j] += upstreamGradient[startIdx + j];
                }
            }
        }
        
        // Update embeddings using embedding optimizer (no weight decay)
        embeddingOptimizer.optimize(embeddings, emptyBiases, embeddingGradients, emptyBiasGradients);
        
        // No gradient to propagate further back
        return null;
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        return backward(stack, stackIndex, upstreamGradient, (ExecutorService) null);
    }
    
    @Override
    public int getOutputSize() {
        return sequenceLength * embeddingDim;
    }
    
    private String[] convertToStringArray(Object input) {
        if (input instanceof String[]) {
            return (String[]) input;
        } else if (input instanceof List) {
            @SuppressWarnings("unchecked")
            List<String> list = (List<String>) input;
            return list.toArray(new String[0]);
        } else {
            throw new IllegalArgumentException(
                "Input must be String[] or List<String>, got: " + 
                (input == null ? "null" : input.getClass().getSimpleName()));
        }
    }
    
    private int getOrAssignTokenId(String word) {
        int tokenId = vocabulary.getIndex(word);
        
        // If vocabulary is full and this is a new word, use <unk>
        if (tokenId >= maxVocabSize) {
            return UNK_INDEX;
        }
        
        return tokenId;
    }
    
    /**
     * Extended context to store the original words for backward pass.
     */
    private static class SequenceLayerContext extends LayerContext {
        final String[] words;
        
        SequenceLayerContext(float[] inputs, float[] weights, float[] outputs, String[] words) {
            super(inputs, weights, outputs);
            this.words = words;
        }
    }
    
    /**
     * Get the current vocabulary size (number of unique words seen).
     */
    public int getVocabularySize() {
        return vocabulary.size();
    }
    
    /**
     * Check if a word is in the vocabulary.
     */
    public boolean hasWord(String word) {
        return vocabulary.containsValue(word);
    }
    
    /**
     * Get embedding for a specific word (for analysis/debugging).
     */
    public float[] getWordEmbedding(String word) {
        int tokenId;
        try {
            tokenId = vocabulary.getIndex(word);
        } catch (IllegalStateException e) {
            // Dictionary is full - use UNK token
            return embeddings[UNK_INDEX].clone();
        }
        if (tokenId == -1 || tokenId >= maxVocabSize) {
            return embeddings[UNK_INDEX].clone(); // Return UNK embedding
        }
        return embeddings[tokenId].clone();
    }
    
    /**
     * Get the token ID for a word. Used by SimpleNetString for tokenization.
     * Returns UNK_INDEX if word is not in vocabulary or vocabulary is full.
     */
    public int getTokenId(String word) {
        int tokenId;
        try {
            tokenId = vocabulary.getIndex(word);
        } catch (IllegalStateException e) {
            // Dictionary is full - use UNK token
            return UNK_INDEX;
        }
        if (tokenId == -1 || tokenId >= maxVocabSize) {
            return UNK_INDEX;
        }
        return tokenId;
    }
    
    /**
     * Get the word for a token ID. Used for converting predictions back to words.
     * Returns <unk> if token ID is invalid.
     */
    public String getWord(int tokenId) {
        if (tokenId < 0 || tokenId >= maxVocabSize)
            return UNK_TOKEN;
        
        Object word = vocabulary.getValue(tokenId);
        return word != null ? word.toString() : UNK_TOKEN;
    }
    
    /**
     * Get the vocabulary dictionary for external use.
     * Note: This is the actual dictionary, not a copy. Use responsibly.
     */
    public Dictionary getVocabulary() {
        return vocabulary;
    }
    
    /**
     * Get the sequence length this layer expects.
     */
    public int getSequenceLength() {
        return sequenceLength;
    }
    
    /**
     * Create a layer specification for a sequence embedding layer.
     */
    public static Layer.Spec spec(int sequenceLength, int maxVocabSize, int embeddingDim, 
                                  Optimizer optimizer, WeightInitStrategy initStrategy) {
        return new InputSequenceEmbeddingLayerSpec(sequenceLength, maxVocabSize, embeddingDim, 
                                                   optimizer, initStrategy, 1.0);
    }
    
    /**
     * Create a layer specification with custom learning rate ratio.
     */
    public static Layer.Spec spec(int sequenceLength, int maxVocabSize, int embeddingDim, 
                                  Optimizer optimizer, WeightInitStrategy initStrategy, 
                                  double learningRateRatio) {
        return new InputSequenceEmbeddingLayerSpec(sequenceLength, maxVocabSize, embeddingDim, 
                                                   optimizer, initStrategy, learningRateRatio);
    }
    
    /**
     * Specification for creating sequence embedding layers.
     */
    private static class InputSequenceEmbeddingLayerSpec extends BaseLayerSpec<InputSequenceEmbeddingLayerSpec> {
        private final int sequenceLength;
        private final int maxVocabSize;
        private final int embeddingDim;
        private final WeightInitStrategy initStrategy;
        
        public InputSequenceEmbeddingLayerSpec(int sequenceLength, int maxVocabSize, int embeddingDim,
                                              Optimizer optimizer, WeightInitStrategy initStrategy, 
                                              double learningRateRatio) {
            super(sequenceLength * embeddingDim, optimizer);
            this.sequenceLength = sequenceLength;
            this.maxVocabSize = maxVocabSize;
            this.embeddingDim = embeddingDim;
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer, FastRandom random) {
            // Input size is the sequence length (ignored, we use our configured length)
            return new InputSequenceEmbeddingLayer(effectiveOptimizer, sequenceLength, 
                                                  maxVocabSize, embeddingDim, initStrategy, random);
        }
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        // Write layer configuration
        out.writeInt(sequenceLength);
        out.writeInt(maxVocabSize);
        out.writeInt(embeddingDim);
        
        // Write vocabulary
        out.writeInt(vocabulary.size());
        for (int i = 0; i < vocabulary.size(); i++) {
            Object value = vocabulary.getValue(i);
            String word = value != null ? value.toString() : "";
            out.writeUTF(word);
        }
        
        // Write embeddings
        for (int i = 0; i < maxVocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                out.writeFloat(embeddings[i][j]);
            }
        }
        
        // Write optimizer
        writeOptimizer(out, optimizer, version);
        
        // Write embedding optimizer (for versions that support it)
        if (embeddingOptimizer != optimizer) {
            out.writeBoolean(true); // Has separate embedding optimizer
            writeOptimizer(out, embeddingOptimizer, version);
        } else {
            out.writeBoolean(false); // No separate embedding optimizer
        }
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize an InputSequenceEmbeddingLayer from stream.
     */
    public static InputSequenceEmbeddingLayer deserialize(DataInputStream in, int version, FastRandom random) throws IOException {
        // Read layer configuration
        int sequenceLength = in.readInt();
        int maxVocabSize = in.readInt();
        int embeddingDim = in.readInt();
        
        // Read vocabulary
        int vocabSize = in.readInt();
        List<String> words = new ArrayList<>();
        for (int i = 0; i < vocabSize; i++) {
            String word = in.readUTF();
            if (!word.isEmpty()) {
                words.add(word);
            }
        }
        
        // Read embeddings
        float[][] embeddings = new float[maxVocabSize][embeddingDim];
        for (int i = 0; i < maxVocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                embeddings[i][j] = in.readFloat();
            }
        }
        
        // Read optimizer
        Optimizer optimizer = readOptimizer(in, version);
        
        // Read embedding optimizer (if separate)
        Optimizer embeddingOptimizer = optimizer;
        boolean hasSeparateEmbeddingOptimizer = in.readBoolean();
        if (hasSeparateEmbeddingOptimizer) {
            embeddingOptimizer = readOptimizer(in, version);
        }
        
        // Create layer with provided FastRandom and restore state
        InputSequenceEmbeddingLayer layer = new InputSequenceEmbeddingLayer(
            optimizer, sequenceLength, maxVocabSize, embeddingDim, WeightInitStrategy.XAVIER, random);
        
        // Override embedding optimizer if different from deserialization
        if (hasSeparateEmbeddingOptimizer) {
            layer.embeddingOptimizer = embeddingOptimizer;
        }
        
        // Restore vocabulary - ensure words are added in order
        for (String word : words) {
            try {
                layer.vocabulary.getIndex(word);
            } catch (IllegalStateException e) {
                // Dictionary is full - skip remaining words
                break;
            }
        }
        
        // Restore embeddings
        for (int i = 0; i < maxVocabSize; i++) {
            System.arraycopy(embeddings[i], 0, layer.embeddings[i], 0, embeddingDim);
        }
        
        return layer;
    }
    
    private static void writeOptimizer(DataOutputStream out, Optimizer optimizer, int version) throws IOException {
        String registeredName = SerializationRegistry.getRegisteredName(optimizer);
        if (registeredName != null) {
            out.writeInt(SerializationConstants.TYPE_CUSTOM);
            out.writeUTF(registeredName);
            return;
        }
        
        Serializable serializableOptimizer = (Serializable) optimizer;
        out.writeInt(serializableOptimizer.getTypeId());
        serializableOptimizer.writeTo(out, version);
    }
    
    private static Optimizer readOptimizer(DataInputStream in, int version) throws IOException {
        int typeId = in.readInt();
        
        if (typeId == SerializationConstants.TYPE_CUSTOM) {
            String className = in.readUTF();
            return SerializationRegistry.createOptimizer(className, in, version);
        }
        
        return switch (typeId) {
            case SerializationConstants.TYPE_SGD_OPTIMIZER -> SgdOptimizer.deserialize(in, version);
            case SerializationConstants.TYPE_ADAM_OPTIMIZER -> AdamOptimizer.deserialize(in, version);
            case SerializationConstants.TYPE_ADAMW_OPTIMIZER -> AdamWOptimizer.deserialize(in, version);
            default -> throw new IOException("Unknown optimizer type ID: " + typeId);
        };
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 12; // sequenceLength + maxVocabSize + embeddingDim
        size += 4; // vocabulary size
        size += vocabulary.size() * 20; // Estimate for UTF strings
        size += maxVocabSize * embeddingDim * 4; // embeddings
        
        // Optimizer size
        String registeredOptimizerName = SerializationRegistry.getRegisteredName(optimizer);
        if (registeredOptimizerName != null) {
            size += 4; // TYPE_CUSTOM
            size += 2 + registeredOptimizerName.getBytes().length; // UTF string
        } else {
            size += 4; // built-in type ID
            size += ((Serializable) optimizer).getSerializedSize(version); // optimizer data
        }
        
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_INPUT_SEQUENCE_EMBEDDING_LAYER;
    }
}