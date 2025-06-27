package dev.neuronic.net.layers;

import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test using InputEmbeddingLayer in a simple language model setup.
 */
class EmbeddingLanguageModelTest {
    
    @Test
    void testSimpleLanguageModel() {
        // Create a simple language model: Single token embedding -> Dense -> Softmax
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        // Vocab: [PAD, THE, CAT, SAT, MAT] = [0, 1, 2, 3, 4]
        int vocabSize = 5;
        int embeddingDim = 8;
        int hiddenSize = 6;
        
        // For this test, use single token input (sequence length = 1)
        NeuralNet languageModel = NeuralNet.newBuilder()
            .input(1) // Single token input
            .layer(Layers.inputEmbedding(vocabSize, embeddingDim, optimizer))
            .layer(Layers.hiddenDenseRelu(hiddenSize, optimizer))
            .output(SoftmaxCrossEntropyOutput.spec(vocabSize, optimizer, WeightInitStrategy.XAVIER));
        
        // Test with single token: "THE" = [1]
        float[] input = {1.0f};
        float[] output = languageModel.predict(input);
        
        assertEquals(vocabSize, output.length, "Output should be vocab-sized probability distribution");
        
        // Output should be a probability distribution (sums to ~1.0)
        float sum = 0.0f;
        for (float prob : output) {
            sum += prob;
            assertTrue(prob >= 0.0f, "Probabilities should be non-negative");
        }
        assertEquals(1.0f, sum, 0.1f, "Probabilities should sum to ~1.0");
    }
    
    @Test
    void testEmbeddingTraining() {
        SgdOptimizer optimizer = new SgdOptimizer(0.1f); // High learning rate for visible changes
        
        // Simple 3-token vocab model with single token input
        NeuralNet model = NeuralNet.newBuilder()
            .input(1) // Single token input
            .layer(Layers.inputEmbedding(3, 4, optimizer))
            .output(SoftmaxCrossEntropyOutput.spec(3, optimizer, WeightInitStrategy.XAVIER));
        
        // Train on simple pattern: [1] -> class 2
        float[] input = {1.0f};
        float[] target = {0.0f, 0.0f, 1.0f}; // One-hot for class 2
        
        // Get initial prediction
        float[] initialPrediction = model.predict(input).clone();
        
        // Train for several iterations
        for (int i = 0; i < 20; i++) {
            model.train(input, target);
        }
        
        // Get final prediction
        float[] finalPrediction = model.predict(input);
        
        // The model should have learned to predict class 2 more strongly
        assertTrue(finalPrediction[2] > initialPrediction[2], 
            "Model should improve prediction for target class");
        
        // Final prediction should prefer class 2
        int predictedClass = 0;
        float maxProb = finalPrediction[0];
        for (int i = 1; i < finalPrediction.length; i++) {
            if (finalPrediction[i] > maxProb) {
                maxProb = finalPrediction[i];
                predictedClass = i;
            }
        }
        assertEquals(2, predictedClass, "Model should predict class 2 after training");
    }
    
    @Test
    void testVariableSequenceLengths() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        // Create embedding layer separately to test different sequence lengths
        InputEmbeddingLayer embeddingLayer = new InputEmbeddingLayer(optimizer, 10, 5, WeightInitStrategy.XAVIER);
        
        // Test different sequence lengths
        for (int seqLen = 1; seqLen <= 10; seqLen++) {
            float[] tokens = new float[seqLen];
            for (int i = 0; i < seqLen; i++) {
                tokens[i] = i % 10; // Valid token IDs
            }
            
            Layer.LayerContext context = embeddingLayer.forward(tokens);
            assertEquals(seqLen * 5, context.outputs().length,
                "Output size should scale with sequence length: " + seqLen);
        }
    }
    
    @Test
    void testEmbeddingLayerInSerialization() {
        // Test that embedding layers work with our serialization system
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        InputEmbeddingLayer layer = new InputEmbeddingLayer(optimizer, 100, 64, WeightInitStrategy.XAVIER);
        
        // Should have correct type ID for serialization
        assertEquals(SerializationConstants.TYPE_INPUT_EMBEDDING_LAYER,
                    layer.getTypeId(), "Should have correct type ID for serialization");
        
        // Should implement Serializable
        assertTrue(layer instanceof Serializable,
                  "Should implement Serializable interface");
    }
    
    @Test
    void testEmbeddingLayerProperties() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        // Test layer created through Layers utility
        Layer.Spec spec = Layers.inputEmbedding(50000, 512, optimizer);
        Layer layer = spec.create(999); // Input size ignored for embeddings
        
        assertTrue(layer instanceof InputEmbeddingLayer, "Should create InputEmbeddingLayer");
        
        InputEmbeddingLayer embeddingLayer = (InputEmbeddingLayer) layer;
        assertEquals(50000, embeddingLayer.getVocabSize(), "Vocab size should match");
        assertEquals(512, embeddingLayer.getEmbeddingDim(), "Embedding dim should match");
        assertEquals(512, embeddingLayer.getOutputSize(), "Output size should be embedding dim");
        assertEquals(512, spec.getOutputSize(), "Spec output size should match layer");
    }
}