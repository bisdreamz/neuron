package dev.neuronic.net.layers;

import dev.neuronic.net.Shape;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationRegistry;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.ExecutorService;

/**
 * Input embedding layer for natural language processing and sequence models.
 * 
 * <p><b>What it does:</b> Converts discrete tokens (words, subwords, characters) into dense vector 
 * representations that neural networks can process. Each unique token gets its own learnable vector.
 * 
 * <p><b>When to use:</b>
 * <ul>
 *   <li><b>Language models:</b> Converting text tokens to embeddings (GPT, BERT, etc.)</li>
 *   <li><b>Sequence models:</b> Processing any discrete sequence data</li>
 *   <li><b>Recommendation systems:</b> Converting item/user IDs to dense representations</li>
 *   <li><b>Time series:</b> Converting categorical time features (day of week, etc.)</li>
 * </ul>
 * 
 * <p><b>How it works:</b>
 * <ol>
 *   <li>Maintains a lookup table: [vocabulary_size × embedding_dimension]</li>
 *   <li>Input: sequence of integer token IDs (e.g., [42, 1337, 256])</li>
 *   <li>Output: concatenated embeddings for each token in sequence</li>
 *   <li>During training: gradients flow back to update embedding vectors</li>
 * </ol>
 * 
 * <p><b>Key Benefits:</b>
 * <ul>
 *   <li><b>Efficient:</b> Simple lookup, no matrix multiplication needed</li>
 *   <li><b>Learnable:</b> Embeddings automatically learn meaningful representations</li>
 *   <li><b>Flexible:</b> Handles variable-length sequences</li>
 *   <li><b>Memory-efficient:</b> Shared embeddings for repeated tokens</li>
 * </ul>
 * 
 * <p><b>Example Usage:</b>
 * <pre>{@code
 * // Setup: 50,000 vocabulary (words/tokens), 512-dimensional embeddings
 * AdamOptimizer optimizer = new AdamOptimizer(0.001f);
 * InputEmbeddingLayer embeddings = new InputEmbeddingLayer(
 *     optimizer, 50000, 512, WeightInitStrategy.XAVIER);
 * 
 * // Example sentence: "Hello world !" -> tokenized as [1234, 5678, 9012]
 * float[] tokenSequence = {1234.0f, 5678.0f, 9012.0f};
 * LayerContext result = embeddings.forward(tokenSequence);
 * 
 * // Output: 3 tokens × 512 dimensions = 1536-dimensional vector
 * float[] embeddings = result.outputs(); // [1536] concatenated embeddings
 * 
 * // Each token's embedding is accessible:
 * // Token 1234: embeddings[0..511]
 * // Token 5678: embeddings[512..1023]  
 * // Token 9012: embeddings[1024..1535]
 * }</pre>
 * 
 * <p><b>Memory Usage:</b> vocabulary_size × embedding_dimension × 4 bytes
 * <br>Example: 50,000 vocab × 512 dims = ~100MB
 * 
 * <p><b>Important Notes:</b>
 * <ul>
 *   <li>Token IDs must be in range [0, vocabulary_size-1]</li>
 *   <li>Use smaller embedding dimensions (128-512) for most applications</li>
 *   <li>Larger vocabularies need more embedding dimensions for good representation</li>
 *   <li>Thread-safe for concurrent training and inference</li>
 * </ul>
 */
public class InputEmbeddingLayer implements Layer, Serializable {
    
    private final Optimizer optimizer;
    private final float[][] embeddings; // [vocabSize][embeddingDim]
    private final int vocabSize;
    private final int embeddingDim;
    private final ThreadLocal<float[]> outputBuffers;
    private final ThreadLocal<float[][]> embeddingGradientBuffers;
    
    public InputEmbeddingLayer(Optimizer optimizer, int vocabSize, int embeddingDim, WeightInitStrategy initStrategy) {
        if (vocabSize <= 0)
            throw new IllegalArgumentException("Vocab size must be positive: " + vocabSize);
        if (embeddingDim <= 0)
            throw new IllegalArgumentException("Embedding dim must be positive: " + embeddingDim);
        
        this.optimizer = optimizer;
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.embeddings = new float[vocabSize][embeddingDim];
        this.outputBuffers = ThreadLocal.withInitial(() -> new float[embeddingDim * 512]);
        this.embeddingGradientBuffers = ThreadLocal.withInitial(() -> new float[vocabSize][embeddingDim]);
        
        switch (initStrategy) {
            case XAVIER -> NetMath.weightInitXavier(embeddings, embeddingDim, embeddingDim);
            case HE -> NetMath.weightInitHe(embeddings, embeddingDim);
        }
    }
    
    @Override
    public LayerContext forward(float[] input, boolean isTraining) {
        if (input.length == 0)
            throw new IllegalArgumentException("Token sequence cannot be empty");
        
        int requiredSize = input.length * embeddingDim;
        float[] output = outputBuffers.get();
        if (output.length < requiredSize) {
            output = new float[requiredSize];
            outputBuffers.set(output);
        }
        
        
        for (int i = 0; i < input.length; i++) {
            int tokenId = (int) input[i];
            if (tokenId != input[i])
                throw new IllegalArgumentException("Input must contain integer token IDs, got: " + input[i]);
            if (tokenId < 0 || tokenId >= vocabSize)
                throw new IllegalArgumentException("Invalid token ID: " + tokenId + " (vocab size: " + vocabSize + ")");
            
            System.arraycopy(embeddings[tokenId], 0, output, i * embeddingDim, embeddingDim);
        }
        
        // Create result array with exactly the right size
        float[] result = new float[requiredSize];
        System.arraycopy(output, 0, result, 0, requiredSize);
        return new LayerContext(input, null, result);
    }
    
    @Override
    public LayerContext forward(float[] input, ExecutorService executor) {
        return forward(input, false);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient, ExecutorService executor) {
        LayerContext context = stack[stackIndex];
        float[] inputFloats = context.inputs();
        
        if (upstreamGradient.length != inputFloats.length * embeddingDim)
            throw new IllegalArgumentException("Gradient length mismatch: expected " + 
                (inputFloats.length * embeddingDim) + ", got " + upstreamGradient.length);
        
        float[][] embeddingGradients = embeddingGradientBuffers.get();
        
        
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                embeddingGradients[i][j] = 0.0f;
            }
        }
        
        
        for (int i = 0; i < inputFloats.length; i++) {
            int tokenId = (int) inputFloats[i];
            int startIdx = i * embeddingDim;
            
            for (int j = 0; j < embeddingDim; j++) {
                embeddingGradients[tokenId][j] += upstreamGradient[startIdx + j];
            }
        }
        
        optimizer.optimize(embeddings, new float[0], embeddingGradients, new float[0]);
        
        return null;
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        return backward(stack, stackIndex, upstreamGradient, (ExecutorService) null);
    }
    
    @Override
    public int getOutputSize() {
        return embeddingDim; // Size per token
    }
    
    public float[] getEmbedding(int tokenId) {
        if (tokenId < 0 || tokenId >= vocabSize)
            throw new IllegalArgumentException("Invalid token ID: " + tokenId);
        return embeddings[tokenId].clone();
    }
    
    public void setEmbedding(int tokenId, float[] embedding) {
        if (tokenId < 0 || tokenId >= vocabSize)
            throw new IllegalArgumentException("Invalid token ID: " + tokenId);
        if (embedding.length != embeddingDim)
            throw new IllegalArgumentException("Embedding dimension mismatch: expected " + 
                embeddingDim + ", got " + embedding.length);
        System.arraycopy(embedding, 0, embeddings[tokenId], 0, embeddingDim);
    }
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public int getEmbeddingDim() {
        return embeddingDim;
    }
    
    /**
     * Create a layer specification for an input embedding layer.
     * 
     * @param vocabSize vocabulary size
     * @param embeddingDim embedding dimension
     * @param optimizer optimizer for embeddings
     * @param initStrategy weight initialization strategy
     */
    public static Layer.Spec spec(int vocabSize, int embeddingDim, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return new InputEmbeddingLayerSpec(vocabSize, embeddingDim, optimizer, initStrategy, 1.0);
    }
    
    /**
     * Create an embedding layer specification with custom learning rate ratio.
     * 
     * @param vocabSize vocabulary size
     * @param embeddingDim embedding dimension
     * @param optimizer optimizer for this layer (null to use default)
     * @param initStrategy weight initialization strategy
     * @param learningRateRatio learning rate scaling factor (1.0 = normal)
     */
    public static Layer.Spec spec(int vocabSize, int embeddingDim, Optimizer optimizer, 
                                  WeightInitStrategy initStrategy, double learningRateRatio) {
        return new InputEmbeddingLayerSpec(vocabSize, embeddingDim, optimizer, initStrategy, learningRateRatio);
    }
    
    /**
     * Specification for creating embedding layers with optimizer management.
     */
    private static class InputEmbeddingLayerSpec extends BaseLayerSpec<InputEmbeddingLayerSpec> {
        private final int vocabSize;
        private final int embeddingDim;
        private final WeightInitStrategy initStrategy;
        
        public InputEmbeddingLayerSpec(int vocabSize, int embeddingDim, Optimizer optimizer, 
                                        WeightInitStrategy initStrategy, double learningRateRatio) {
            super(embeddingDim, optimizer);
            this.vocabSize = vocabSize;
            this.embeddingDim = embeddingDim;
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        @Override
        public Layer create(int inputSize) {
            return createLayer(inputSize, getEffectiveOptimizer(null));
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            // Input size is ignored for embedding layers (determined by sequence length)
            return new InputEmbeddingLayer(effectiveOptimizer, vocabSize, embeddingDim, initStrategy);
        }
        
        @Override
        public boolean prefersShapeAPI() {
            return true; // Embeddings benefit from knowing they're processing sequences
        }
        
        @Override
        public void validateInputShape(Shape inputShape) {
            // Accept 1D (vector of token IDs) or 2D (batch of sequences)
            if (inputShape.rank() > 2) {
                throw new IllegalArgumentException(
                    "InputEmbedding expects 1D token sequence or 2D batch, got shape: " + inputShape);
            }
        }
        
        @Override
        public Shape getOutputShape(Shape inputShape) {
            if (inputShape.rank() == 1) {
                // [seqLen] -> [seqLen, embeddingDim]
                return Shape.sequence(inputShape.dim(0), embeddingDim);
            } else if (inputShape.rank() == 2 && inputShape.dim(1) == 1) {
                // [seqLen, 1] -> [seqLen, embeddingDim]
                return Shape.sequence(inputShape.dim(0), embeddingDim);
            }
            // Fallback to flat calculation
            return Shape.vector(getOutputSize(inputShape.toFlatSize()));
        }
        
        @Override
        public int getOutputSize(int inputSize) {
            // For sequences: inputSize tokens × embeddingDim
            return inputSize * embeddingDim;
        }
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        // Write layer dimensions
        out.writeInt(vocabSize);
        out.writeInt(embeddingDim);
        
        
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                out.writeFloat(embeddings[i][j]);
            }
        }
        
        // Write optimizer
        writeOptimizer(out, optimizer, version);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize an InputEmbeddingLayer from stream.
     */
    public static InputEmbeddingLayer deserialize(DataInputStream in, int version) throws IOException {
        // Read layer dimensions
        int vocabSize = in.readInt();
        int embeddingDim = in.readInt();
        
        float[][] embeddings = new float[vocabSize][embeddingDim];
        
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                embeddings[i][j] = in.readFloat();
            }
        }
        
        // Read optimizer
        Optimizer optimizer = readOptimizer(in, version);
        
        // Create layer and set deserialized values
        InputEmbeddingLayer layer = new InputEmbeddingLayer(optimizer, vocabSize, embeddingDim, WeightInitStrategy.XAVIER);
        
        
        for (int i = 0; i < vocabSize; i++) {
            System.arraycopy(embeddings[i], 0, layer.embeddings[i], 0, embeddingDim);
        }
        
        return layer;
    }
    
    private static void writeOptimizer(DataOutputStream out, Optimizer optimizer, int version) throws IOException {
        // Check if it's a registered custom optimizer
        String registeredName = SerializationRegistry.getRegisteredName(optimizer);
        if (registeredName != null) {
            out.writeInt(SerializationConstants.TYPE_CUSTOM);
            out.writeUTF(registeredName);
            return;
        }
        
        // Use built-in serialization
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
            default -> throw new IOException("Unknown optimizer type ID: " + typeId);
        };
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 8; // vocabSize + embeddingDim
        size += vocabSize * embeddingDim * 4; // embeddings
        
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
        return SerializationConstants.TYPE_INPUT_EMBEDDING_LAYER;
    }
}