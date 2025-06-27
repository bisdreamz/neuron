package dev.neuronic.net.layers;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.Shape;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.layers.Layer.LayerContext;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GRU layer output modes (ALL_TIMESTEPS vs LAST_TIMESTEP).
 */
public class GruOutputModeTest {
    
    @Test
    public void testGruLastTimestepOutputSize() {
        // Create a simple model with GRU that outputs only last timestep
        int sequenceLength = 10;
        int embeddingDim = 16;
        int hiddenSize = 32;
        
        NeuralNet model = NeuralNet.newBuilder()
                .input(sequenceLength)
                .setDefaultOptimizer(new SgdOptimizer(0.01f))
                .layer(Layers.inputEmbedding(1000, embeddingDim))
                .layer(Layers.hiddenGruLast(hiddenSize))
                .output(Layers.outputSoftmaxCrossEntropy(10));
        
        // Test forward pass
        float[] input = new float[sequenceLength];
        for (int i = 0; i < sequenceLength; i++) {
            input[i] = i % 100; // Some token IDs
        }
        
        float[] output = model.predict(input);
        
        // Output should be 10 classes (from softmax)
        assertEquals(10, output.length);
    }
    
    @Test
    public void testGruAllTimestepsOutputSize() {
        // Create a simple model with GRU that outputs all timesteps
        int sequenceLength = 10;
        int embeddingDim = 16;
        int hiddenSize = 32;
        
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        // Use shape-aware API!
        NeuralNet model = NeuralNet.newBuilder()
                .input(Shape.sequence(sequenceLength, 1))  // Shape-aware input
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputEmbedding(1000, embeddingDim))
                .layer(Layers.hiddenGruAll(hiddenSize))  // No hint needed with shapes!
                // Need to handle all timesteps - add dense layer to reduce
                .layer(Layers.hiddenDenseRelu(64))
                .output(Layers.outputSoftmaxCrossEntropy(10));
        
        // Test forward pass
        float[] input = new float[sequenceLength];
        for (int i = 0; i < sequenceLength; i++) {
            input[i] = i % 100; // Some token IDs
        }
        
        float[] output = model.predict(input);
        
        // Output should be 10 classes (from softmax)
        assertEquals(10, output.length);
    }
    
    @Test
    public void testGruLayerOutputModes() {
        // Test the GRU layer directly
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        int hiddenSize = 8;
        int inputSize = 4;
        int seqLen = 3;
        
        // Test ALL_TIMESTEPS mode
        GruLayer gruAll = new GruLayer(optimizer, hiddenSize, inputSize, 
                                      WeightInitStrategy.XAVIER, GruLayer.OutputMode.ALL_TIMESTEPS);
        
        float[] input = new float[seqLen * inputSize];
        LayerContext contextAll = gruAll.forward(input);
        
        // Should output all timesteps: seqLen * hiddenSize
        assertEquals(seqLen * hiddenSize, contextAll.outputs().length);
        
        // Test LAST_TIMESTEP mode
        GruLayer gruLast = new GruLayer(optimizer, hiddenSize, inputSize, 
                                       WeightInitStrategy.XAVIER, GruLayer.OutputMode.LAST_TIMESTEP);
        
        LayerContext contextLast = gruLast.forward(input);
        
        // Should output only last timestep: hiddenSize
        assertEquals(hiddenSize, contextLast.outputs().length);
    }
    
    @Test
    public void testBackwardCompatibility() {
        // Test that old spec() method still works (defaults to ALL_TIMESTEPS)
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec oldSpec = GruLayer.spec(32, optimizer, WeightInitStrategy.XAVIER);
        
        GruLayer oldGru = (GruLayer) oldSpec.create(16);
        
        // Should behave like ALL_TIMESTEPS by default
        float[] input = new float[5 * 16]; // 5 timesteps, 16 input size
        LayerContext context = oldGru.forward(input);
        
        assertEquals(5 * 32, context.outputs().length); // All timesteps
    }
}