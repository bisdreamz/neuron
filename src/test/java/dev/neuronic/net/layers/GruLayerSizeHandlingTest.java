package dev.neuronic.net.layers;

import dev.neuronic.net.*;
import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that GRU layer correctly handles sequence input sizes.
 */
public class GruLayerSizeHandlingTest {
    
    @Test
    public void testGruInfersCorrectInputSizeFromSequence() {
        // Simulate a language model with embedding layer feeding into GRU
        int seqLen = 20;
        int vocabSize = 10000;
        int embeddingDim = 128;
        int hiddenSize = 128;
        
        NeuralNet model = NeuralNet.newBuilder()
            .input(seqLen)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.inputSequenceEmbedding(seqLen, vocabSize, embeddingDim))
            .layer(Layers.hiddenGruAllNormalized(hiddenSize))
            .layer(Layers.dropout(0.3f))
            .layer(Layers.hiddenGruLastNormalized(hiddenSize))
            .layer(Layers.dropout(0.4f))
            .output(Layers.outputSoftmaxCrossEntropy(vocabSize));
        
        // The model should build without throwing
        assertNotNull(model);
        assertEquals(6, model.getLayers().length); // 5 layers + 1 output
        
        // Test forward pass with dummy input
        float[] input = new float[seqLen];
        for (int i = 0; i < seqLen; i++) {
            input[i] = i; // Token IDs
        }
        
        // Should not throw
        float[] output = model.predict(input);
        assertEquals(vocabSize, output.length);
    }
    
    @Test
    public void testGruHandlesVariousEmbeddingDimensions() {
        // Test with different embedding dimensions
        int[][] configs = {
            {20, 64},   // seqLen=20, embDim=64
            {50, 128},  // seqLen=50, embDim=128
            {100, 256}, // seqLen=100, embDim=256
            {10, 512}   // seqLen=10, embDim=512
        };
        
        for (int[] config : configs) {
            int seqLen = config[0];
            int embDim = config[1];
            int flattenedSize = seqLen * embDim;
            
            // Create GRU spec and layer with expected input dimension
            Layer.Spec gruSpec = GruLayer.specAll(128, new SgdOptimizer(0.01f), WeightInitStrategy.XAVIER, embDim);
            Layer gruLayer = gruSpec.create(flattenedSize);
            
            // Forward pass with flattened sequence
            float[] input = new float[flattenedSize];
            Layer.LayerContext ctx = gruLayer.forward(input);
            
            // Output should be seqLen * hiddenSize
            assertEquals(seqLen * 128, ctx.outputs().length,
                "For seqLen=" + seqLen + ", embDim=" + embDim);
        }
    }
    
    @Test
    public void testDropoutHandlesDynamicSizes() {
        // Test dropout with different input sizes
        DropoutLayer dropout = new DropoutLayer(0.5f);
        
        // Should handle any size
        int[] sizes = {128, 256, 2560, 20 * 128};
        for (int size : sizes) {
            float[] input = new float[size];
            Layer.LayerContext ctx = dropout.forward(input);
            assertEquals(size, ctx.outputs().length);
        }
    }
}