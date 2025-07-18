package dev.neuronic.net.layers;

import dev.neuronic.net.*;
import dev.neuronic.net.optimizers.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the inputSequenceNumerical layer and its interaction with GRU layers.
 */
class SequenceNumericalInputLayerTest {
    
    private AdamWOptimizer optimizer;
    
    @BeforeEach
    void setUp() {
        optimizer = new AdamWOptimizer(0.001f, 0.01f);
    }
    
    @Test
    void testSequenceNumericalWithGru() {
        // This should work - proper sequence input
        NeuralNet model = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(30)  // Specify input size
            .layer(Layers.inputSequenceNumerical(30, Feature.autoScale(900f, 1500f)))
            .layer(Layers.hiddenGruLast(16))
            .layer(Layers.hiddenDenseRelu(16))
            .withGlobalGradientClipping(1f)
            .output(Layers.outputLinearRegression(1));
        
        assertNotNull(model);
        
        // Test prediction
        float[] last30Days = new float[30];
        for (int i = 0; i < 30; i++) {
            last30Days[i] = 1000f + i * 10f; // Simulated revenue data
        }
        
        float[] prediction = model.predict(last30Days);
        assertNotNull(prediction);
        assertEquals(1, prediction.length);
        // Output should be a reasonable value (not NaN or infinity)
        assertTrue(Float.isFinite(prediction[0]));
    }
    
    @Test
    void testInputMixedWithGruFailsWithHelpfulError() {
        // This should fail with helpful error message
        Feature[] features = new Feature[30];
        for (int i = 0; i < 30; i++) {
            features[i] = Feature.autoScale(900f, 1500f);
        }
        
        Exception ex = assertThrows(Exception.class, () -> {
            NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenGruAll(16))
                .output(Layers.outputLinearRegression(1));
        });
        
        // Check that error message is helpful
        String message = ex.getMessage();
        assertTrue(message.contains("GRU layer cannot process non-sequence input") ||
                   message.contains("GRU layer requires shape information"),
                   "Error message should guide users to use sequence inputs. Got: " + message);
    }
    
    @Test
    void testSequenceNumericalWithDifferentFeatureTypes() {
        // Test with different scaling options
        
        // Auto-normalize
        NeuralNet model1 = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(20)
            .layer(Layers.inputSequenceNumerical(20, Feature.autoNormalize()))
            .layer(Layers.hiddenGruLast(8))
            .output(Layers.outputLinearRegression(1));
        assertNotNull(model1);
        
        // Passthrough
        NeuralNet model2 = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(20)
            .layer(Layers.inputSequenceNumerical(20, Feature.passthrough()))
            .layer(Layers.hiddenGruLast(8))
            .output(Layers.outputLinearRegression(1));
        assertNotNull(model2);
        
        // Test predictions work
        float[] data = new float[20];
        for (int i = 0; i < 20; i++) {
            data[i] = i * 0.1f;
        }
        
        assertDoesNotThrow(() -> model1.predict(data));
        assertDoesNotThrow(() -> model2.predict(data));
    }
    
    @Test
    void testSequenceNumericalValidatesLength() {
        NeuralNet model = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(10)
            .layer(Layers.inputSequenceNumerical(10, Feature.passthrough()))
            .output(Layers.outputLinearRegression(10));
        
        // Wrong input length should throw
        float[] wrongLength = new float[15];
        assertThrows(IllegalArgumentException.class, () -> model.predict(wrongLength));
    }
}