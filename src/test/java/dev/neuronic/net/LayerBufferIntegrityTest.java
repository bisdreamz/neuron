package dev.neuronic.net;

import dev.neuronic.net.*;
import dev.neuronic.net.layers.*;
import dev.neuronic.net.activators.TanhActivator;
import dev.neuronic.net.layers.*;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.outputs.*;
import dev.neuronic.net.activators.LinearActivator;
import dev.neuronic.net.outputs.LinearRegressionOutput;
import dev.neuronic.net.outputs.MultiLabelSigmoidOutput;
import dev.neuronic.net.outputs.SigmoidBinaryCrossEntropyOutput;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Critical integration tests to detect ThreadLocal buffer corruption bugs.
 * 
 * These tests are designed to catch the exact type of buffer reuse issues that
 * were missed by component-level tests but broke neural network functionality.
 * 
 * The pattern we're testing:
 * 1. Call layer.forward() with input A
 * 2. Save the LayerContext 
 * 3. Call layer.forward() with input B
 * 4. Verify LayerContext from step 2 hasn't been corrupted
 */
public class LayerBufferIntegrityTest {
    
    @Test
    public void testDenseLayerBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        FastRandom random = new FastRandom(12345);
        DenseLayer layer = new DenseLayer(optimizer, TanhActivator.INSTANCE,
                                         3, 2, WeightInitStrategy.XAVIER, random);
        
        float[] inputA = {1.0f, 0.0f};
        float[] inputB = {0.0f, 1.0f};
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        float[] originalPreActivationsA = contextA.preActivations().clone();
        
        // Second call (should not corrupt first context)
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify first context wasn't corrupted
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f, 
            "DenseLayer: First context outputs corrupted by second call");
        assertArrayEquals(originalPreActivationsA, contextA.preActivations(), 0.001f,
            "DenseLayer: First context preActivations corrupted by second call");
        
        // Verify outputs are actually different  
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "DenseLayer: Different inputs should produce different outputs");
    }
    
    @Test
    public void testGruLayerBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        GruLayer layer = new GruLayer(optimizer, 4, 3, WeightInitStrategy.XAVIER, new FastRandom(12345));
        
        float[] inputA = {1.0f, 0.0f, 0.5f};
        float[] inputB = {0.0f, 1.0f, -0.5f};
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        // Note: GRU intentionally returns null for preActivations, so we skip that check
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption of outputs
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "GruLayer: First context outputs corrupted by second call");
        
        // Verify different outputs
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "GruLayer: Different inputs should produce different outputs");
    }
    
    @Test
    public void testInputSequenceEmbeddingLayerBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        InputSequenceEmbeddingLayer layer = new InputSequenceEmbeddingLayer(
            optimizer, 3, 10, 4, WeightInitStrategy.XAVIER, new FastRandom(12345));
        
        // Build vocabulary first
        layer.getTokenId("a"); // ID 1
        layer.getTokenId("b"); // ID 2
        
        float[] inputA = {1.0f, 1.0f, 1.0f}; // "a a a"
        float[] inputB = {2.0f, 2.0f, 2.0f}; // "b b b"
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "InputSequenceEmbeddingLayer: First context outputs corrupted by second call");
        
        // Verify different outputs
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "InputSequenceEmbeddingLayer: Different inputs should produce different outputs");
    }
    
    @Test
    public void testSoftmaxCrossEntropyOutputBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        SoftmaxCrossEntropyOutput layer = new SoftmaxCrossEntropyOutput(
            optimizer, 3, 2, WeightInitStrategy.XAVIER, new FastRandom(12345));
        
        float[] inputA = {1.0f, 0.0f};
        float[] inputB = {0.0f, 1.0f};
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        float[] originalPreActivationsA = contextA.preActivations().clone();
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "SoftmaxCrossEntropyOutput: First context outputs corrupted by second call");
        assertArrayEquals(originalPreActivationsA, contextA.preActivations(), 0.001f,
            "SoftmaxCrossEntropyOutput: First context preActivations corrupted by second call");
        
        // Verify different outputs
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "SoftmaxCrossEntropyOutput: Different inputs should produce different outputs");
    }
    
    @Test
    public void testLinearRegressionOutputBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        LinearRegressionOutput layer = new LinearRegressionOutput(optimizer, 2, 3, new FastRandom(12345));
        
        float[] inputA = {1.0f, 0.0f, 0.5f};
        float[] inputB = {0.0f, 1.0f, -0.5f};
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        float[] originalPreActivationsA = contextA.preActivations().clone();
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "LinearRegressionOutput: First context outputs corrupted by second call");
        assertArrayEquals(originalPreActivationsA, contextA.preActivations(), 0.001f,
            "LinearRegressionOutput: First context preActivations corrupted by second call");
        
        // Verify different outputs
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "LinearRegressionOutput: Different inputs should produce different outputs");
    }
    
    @Test
    public void testDropoutLayerBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        DropoutLayer layer = new DropoutLayer(0.1f, new FastRandom(12345)); // Low dropout for predictable testing
        
        float[] inputA = {1.0f, 2.0f, 3.0f};
        float[] inputB = {4.0f, 5.0f, 6.0f};
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "DropoutLayer: First context outputs corrupted by second call");
        
        // Note: With dropout always active, outputs won't be identical to inputs
        // Just verify that contexts maintain their own state
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "DropoutLayer: Different inputs should produce different outputs");
    }
    
    @Test
    public void testLayerNormLayerBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        LayerNormLayer layer = new LayerNormLayer(optimizer, 3);
        
        // Use inputs with different distributions to ensure different normalized outputs
        float[] inputA = {10.0f, 20.0f, 30.0f}; // mean=20, std=8.16  
        float[] inputB = {1.0f, 1.0f, 10.0f};   // mean=4, std=4.24
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "LayerNormLayer: First context outputs corrupted by second call");
        
        // Verify different outputs (even with layer norm, different distributions should produce different outputs)
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "LayerNormLayer: Different inputs should produce different outputs");
    }
    
    @Test
    public void testMultipleSequentialCalls() {
        // Test even more aggressive buffer reuse scenarios
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        FastRandom random = new FastRandom(12345);
        DenseLayer layer = new DenseLayer(optimizer, LinearActivator.INSTANCE,
                                         2, 2, WeightInitStrategy.XAVIER, random);
        
        // Make many calls and save all contexts
        Layer.LayerContext[] contexts = new Layer.LayerContext[5];
        float[][] expectedOutputs = new float[5][];
        
        for (int i = 0; i < 5; i++) {
            float[] input = {i * 1.0f, i * 2.0f};
            contexts[i] = layer.forward(input, false);
            expectedOutputs[i] = contexts[i].outputs().clone();
        }
        
        // Verify all contexts remain uncorrupted
        for (int i = 0; i < 5; i++) {
            assertArrayEquals(expectedOutputs[i], contexts[i].outputs(), 0.001f,
                "Context " + i + " was corrupted by subsequent calls");
        }
        
        // Verify all outputs are different
        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 5; j++) {
                assertFalse(java.util.Arrays.equals(contexts[i].outputs(), contexts[j].outputs()),
                    "Contexts " + i + " and " + j + " should have different outputs");
            }
        }
    }
    
    @Test
    public void testSigmoidBinaryCrossEntropyOutputBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        FastRandom random = new FastRandom(12345);
        SigmoidBinaryCrossEntropyOutput layer = new SigmoidBinaryCrossEntropyOutput(optimizer, 3, random);
        
        float[] inputA = {1.0f, 0.0f, 0.5f};
        float[] inputB = {0.0f, 1.0f, -0.5f};
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        float[] originalPreActivationsA = contextA.preActivations().clone();
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "SigmoidBinaryCrossEntropyOutput: First context outputs corrupted by second call");
        assertArrayEquals(originalPreActivationsA, contextA.preActivations(), 0.001f,
            "SigmoidBinaryCrossEntropyOutput: First context preActivations corrupted by second call");
        
        // Verify different outputs
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "SigmoidBinaryCrossEntropyOutput: Different inputs should produce different outputs");
    }
    
    @Test
    public void testMultiLabelSigmoidOutputBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        FastRandom random = new FastRandom(12345);
        MultiLabelSigmoidOutput layer = new MultiLabelSigmoidOutput(optimizer, 3, 2, random);
        
        float[] inputA = {1.0f, 0.0f};
        float[] inputB = {0.0f, 1.0f};
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        float[] originalPreActivationsA = contextA.preActivations().clone();
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "MultiLabelSigmoidOutput: First context outputs corrupted by second call");
        assertArrayEquals(originalPreActivationsA, contextA.preActivations(), 0.001f,
            "MultiLabelSigmoidOutput: First context preActivations corrupted by second call");
        
        // Verify different outputs
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "MultiLabelSigmoidOutput: Different inputs should produce different outputs");
    }
    
    @Test
    public void testInputEmbeddingLayerBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        FastRandom random = new FastRandom(12345);
        InputEmbeddingLayer layer = new InputEmbeddingLayer(optimizer, 100, 32, WeightInitStrategy.XAVIER, random);
        
        float[] inputA = {1.0f, 5.0f, 10.0f}; // Token IDs
        float[] inputB = {2.0f, 8.0f, 15.0f}; // Different token IDs
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "InputEmbeddingLayer: First context outputs corrupted by second call");
        
        // Verify different outputs
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "InputEmbeddingLayer: Different inputs should produce different outputs");
    }
    
    @Test
    public void testMixedFeatureInputLayerBufferIntegrity() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        Feature[] features = {
            Feature.embedding(100, 32),  // Categorical feature with 100 possible values
            Feature.oneHot(10),          // Categorical with 10 categories
            Feature.passthrough(),       // Numerical feature
            Feature.autoNormalize()      // Auto-normalized numerical feature
        };
        FastRandom random = new FastRandom(12345);
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER, random);
        
        float[] inputA = {1.0f, 5.0f, 0.5f, 1.5f}; // categorical + continuous features
        float[] inputB = {2.0f, 8.0f, -0.5f, 2.5f}; // Different features
        
        // First call
        Layer.LayerContext contextA = layer.forward(inputA, false);
        float[] originalOutputsA = contextA.outputs().clone();
        
        // Second call
        Layer.LayerContext contextB = layer.forward(inputB, false);
        
        // Verify no corruption
        assertArrayEquals(originalOutputsA, contextA.outputs(), 0.001f,
            "MixedFeatureInputLayer: First context outputs corrupted by second call");
        
        // Verify different outputs
        assertFalse(java.util.Arrays.equals(contextA.outputs(), contextB.outputs()),
            "MixedFeatureInputLayer: Different inputs should produce different outputs");
    }
    
    
    @Test
    public void testNetworkLevelBufferIntegrity() {
        // Test the exact scenario that was broken: full neural network with multiple layers
        // Using Tanh activation to avoid dead ReLU issues that can cause identical outputs
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
            .layer(Layers.hiddenDenseTanh(4))
            .layer(Layers.hiddenDenseTanh(3))
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        float[] inputA = {1.0f, 0.0f};
        float[] inputB = {0.0f, 1.0f};
        float[] inputC = {0.5f, 0.5f};
        
        // Multiple predictions
        float[] outputA = net.predict(inputA);
        float[] outputB = net.predict(inputB);
        float[] outputC = net.predict(inputC);
        
        // All should be different
        assertFalse(java.util.Arrays.equals(outputA, outputB),
            "Network: Inputs A and B should produce different outputs");
        assertFalse(java.util.Arrays.equals(outputB, outputC),
            "Network: Inputs B and C should produce different outputs");
        assertFalse(java.util.Arrays.equals(outputA, outputC),
            "Network: Inputs A and C should produce different outputs");
        
        // Verify outputs are valid probabilities (sum to 1)
        float sumA = 0, sumB = 0, sumC = 0;
        for (float v : outputA) sumA += v;
        for (float v : outputB) sumB += v;
        for (float v : outputC) sumC += v;
        
        assertEquals(1.0f, sumA, 0.01f, "Network output A should sum to 1 (softmax)");
        assertEquals(1.0f, sumB, 0.01f, "Network output B should sum to 1 (softmax)");
        assertEquals(1.0f, sumC, 0.01f, "Network output C should sum to 1 (softmax)");
    }
}