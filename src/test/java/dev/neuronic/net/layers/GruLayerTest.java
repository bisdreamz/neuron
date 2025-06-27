package dev.neuronic.net.layers;

import dev.neuronic.net.Layers;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.serialization.SerializationConstants;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test for GRU layer functionality.
 */
class GruLayerTest {
    
    @Test
    void testBasicGruForward() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 4, 3, WeightInitStrategy.XAVIER);
        
        // Test with sequence of length 2
        float[] input = {
            1.0f, 0.5f, 0.0f,  // timestep 1: [1.0, 0.5, 0.0]
            0.0f, 1.0f, 0.5f   // timestep 2: [0.0, 1.0, 0.5]
        };
        
        Layer.LayerContext context = gru.forward(input);
        float[] output = context.outputs();
        
        assertEquals(2 * 4, output.length, "Output should be seqLen * hiddenSize");
        assertEquals(4, gru.getOutputSize(), "Output size should be hiddenSize");
        assertEquals(4, gru.getHiddenSize(), "Hidden size should match");
        assertEquals(3, gru.getInputSize(), "Input size should match");
        
        // Output should be bounded (tanh activation in final step)
        for (float val : output) {
            assertTrue(val >= -1.0f && val <= 1.0f, "GRU output should be bounded by tanh: " + val);
        }
    }
    
    @Test
    void testGruDimensions() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 8, 5, WeightInitStrategy.HE);
        
        // Test different sequence lengths
        for (int seqLen = 1; seqLen <= 5; seqLen++) {
            float[] input = new float[seqLen * 5];
            for (int i = 0; i < input.length; i++) {
                input[i] = (float) Math.random();
            }
            
            Layer.LayerContext context = gru.forward(input);
            assertEquals(seqLen * 8, context.outputs().length,
                "Output size should scale with sequence length: " + seqLen);
        }
    }
    
    @Test
    void testGruStateful() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 3, 2, WeightInitStrategy.XAVIER);
        
        // Same input should produce same output (deterministic)
        float[] input = {1.0f, 0.0f, 0.5f, 1.0f}; // 2 timesteps
        
        float[] output1 = gru.forward(input).outputs();
        float[] output2 = gru.forward(input).outputs();
        
        assertArrayEquals(output1, output2, 1e-6f, "GRU should be deterministic");
        
        // Different inputs should produce different outputs
        float[] differentInput = {0.0f, 1.0f, 1.0f, 0.5f};
        float[] output3 = gru.forward(differentInput).outputs();
        
        assertFalse(java.util.Arrays.equals(output1, output3), "Different inputs should produce different outputs");
    }
    
    @Test
    void testGruSerialization() throws IOException {
        SgdOptimizer optimizer = new SgdOptimizer(0.02f);
        GruLayer original = new GruLayer(optimizer, 4, 3, WeightInitStrategy.XAVIER);
        
        // Test forward pass
        float[] input = {1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.5f};
        float[] originalOutput = original.forward(input).outputs();
        
        // Serialize
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(baos);
        original.writeTo(out, SerializationConstants.CURRENT_VERSION);
        out.close();
        
        // Deserialize
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        DataInputStream in = new DataInputStream(bais);
        GruLayer deserialized = GruLayer.deserialize(in, SerializationConstants.CURRENT_VERSION);
        in.close();
        
        // Test equivalence
        assertEquals(original.getHiddenSize(), deserialized.getHiddenSize());
        assertEquals(original.getInputSize(), deserialized.getInputSize());
        assertEquals(original.getTypeId(), deserialized.getTypeId());
        
        // Test forward pass equivalence
        float[] deserializedOutput = deserialized.forward(input).outputs();
        assertArrayEquals(originalOutput, deserializedOutput, 1e-6f,
            "Forward pass should be identical after serialization");
    }
    
    @Test
    void testGruLayerSpec() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.gru(16, optimizer);
        
        assertEquals(16, spec.getOutputSize(), "Spec should return hidden size");
        
        Layer layer = spec.create(8); // inputSize = 8
        assertTrue(layer instanceof GruLayer, "Should create GruLayer");
        
        GruLayer gruLayer = (GruLayer) layer;
        assertEquals(16, gruLayer.getHiddenSize(), "Hidden size should match spec");
        assertEquals(8, gruLayer.getInputSize(), "Input size should match create parameter");
    }
    
    @Test
    void testInvalidInputs() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 4, 3, WeightInitStrategy.XAVIER);
        
        // Input length not multiple of inputSize
        assertThrows(IllegalArgumentException.class, () -> {
            gru.forward(new float[]{1.0f, 2.0f}); // length=2, inputSize=3
        });
        
        // Input with wrong dimensions should work but might produce unexpected results
        // Empty input will have seqLen=0, which should still be valid
    }
    
    @Test
    void testConstructorValidation() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        // Invalid hidden size
        assertThrows(IllegalArgumentException.class, () -> {
            new GruLayer(optimizer, 0, 10, WeightInitStrategy.XAVIER);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new GruLayer(optimizer, -5, 10, WeightInitStrategy.XAVIER);
        });
        
        // Invalid input size
        assertThrows(IllegalArgumentException.class, () -> {
            new GruLayer(optimizer, 10, 0, WeightInitStrategy.XAVIER);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new GruLayer(optimizer, 10, -3, WeightInitStrategy.XAVIER);
        });
    }
    
    @Test
    void testBackwardPass() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 4, 3, WeightInitStrategy.XAVIER);
        
        float[] input = {1.0f, 0.0f, 0.5f};
        Layer.LayerContext context = gru.forward(input);
        
        // Backward pass should now work
        float[] upstreamGrad = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] inputGradients = gru.backward(new Layer.LayerContext[]{context}, 0, upstreamGrad);
        
        assertNotNull(inputGradients, "Input gradients should not be null");
        assertEquals(3, inputGradients.length, "Input gradients should match input size");
        
        // Gradients should be finite numbers
        for (float grad : inputGradients) {
            assertTrue(Float.isFinite(grad), "All gradients should be finite: " + grad);
        }
    }
    
    @Test 
    void testBackwardPassMultipleTimesteps() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 3, 2, WeightInitStrategy.XAVIER);
        
        // Test with 3 timesteps
        float[] input = {
            1.0f, 0.5f,  // t=0
            0.0f, 1.0f,  // t=1  
            0.5f, 0.0f   // t=2
        };
        Layer.LayerContext context = gru.forward(input);
        
        // Upstream gradients for all timesteps
        float[] upstreamGrad = {
            0.1f, 0.2f, 0.3f,  // t=0 gradients
            0.4f, 0.5f, 0.6f,  // t=1 gradients
            0.7f, 0.8f, 0.9f   // t=2 gradients
        };
        
        float[] inputGradients = gru.backward(new Layer.LayerContext[]{context}, 0, upstreamGrad);
        
        assertNotNull(inputGradients, "Input gradients should not be null");
        assertEquals(6, inputGradients.length, "Input gradients should match sequence length * input size");
        
        // Gradients should be finite and non-zero for meaningful learning
        boolean hasNonZeroGrad = false;
        for (float grad : inputGradients) {
            assertTrue(Float.isFinite(grad), "All gradients should be finite: " + grad);
            if (Math.abs(grad) > 1e-6f) {
                hasNonZeroGrad = true;
            }
        }
        assertTrue(hasNonZeroGrad, "Should have some non-zero gradients for learning");
    }
    
    @Test
    void testGruLearning() {
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        GruLayer gru = new GruLayer(optimizer, 2, 1, WeightInitStrategy.XAVIER);
        
        // Simple learning test: try to learn to output [1.0, 0.0] for input [1.0]
        float[] input = {1.0f};
        float[] target = {1.0f, 0.0f};
        
        // Get initial output
        Layer.LayerContext context1 = gru.forward(input);
        float[] output1 = context1.outputs().clone();
        
        // Compute simple MSE gradients
        float[] gradients = new float[2];
        for (int i = 0; i < 2; i++) {
            gradients[i] = 2.0f * (output1[i] - target[i]); // d/dx(x-target)² = 2(x-target)
        }
        
        // Backward pass (updates weights)
        gru.backward(new Layer.LayerContext[]{context1}, 0, gradients);
        
        // Forward pass again to see if it learned
        Layer.LayerContext context2 = gru.forward(input);
        float[] output2 = context2.outputs();
        
        // Output should be different after learning (weights updated)
        boolean outputChanged = false;
        for (int i = 0; i < 2; i++) {
            if (Math.abs(output1[i] - output2[i]) > 1e-6f) {
                outputChanged = true;
                break;
            }
        }
        assertTrue(outputChanged, "Output should change after learning step");
    }
}