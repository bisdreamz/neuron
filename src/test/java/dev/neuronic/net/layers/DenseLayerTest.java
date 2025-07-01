package dev.neuronic.net.layers;

import dev.neuronic.net.activators.Activator;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.WeightInitStrategy;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DenseLayerTest {
    
    private Activator mockActivator;
    private Optimizer mockOptimizer;
    
    @BeforeEach
    void setUp() {
        // Simple identity activator for testing
        mockActivator = new Activator() {
            @Override
            public void activate(float[] input, float[] output) {
                System.arraycopy(input, 0, output, 0, input.length);
            }
            
            @Override
            public void derivative(float[] input, float[] output) {
                for (int i = 0; i < output.length; i++)
                    output[i] = 1.0f;
            }
        };
        
        // No-op optimizer for testing
        mockOptimizer = new Optimizer() {
            @Override
            public void optimize(float[][] weights, float[] biases, 
                               float[][] weightGradients, float[] biasGradients) {
                // No-op
            }
            
            @Override
            public void setLearningRate(float learningRate) {
                // No-op for test
            }
        };
    }
    
    @Test
    void testDenseLayerConstruction() {
        DenseLayer layer = new DenseLayer(mockOptimizer, mockActivator, 3, 2, WeightInitStrategy.HE);
        assertEquals(3, layer.getOutputSize());
    }
    
    @Test
    void testForwardPropagation() {
        DenseLayer layer = new DenseLayer(mockOptimizer, mockActivator, 3, 2, WeightInitStrategy.HE);
        float[] input = {1.0f, 2.0f};
        
        Layer.LayerContext context = layer.forward(input, false);
        
        assertNotNull(context);
        assertArrayEquals(input, context.inputs());
        assertEquals(3, context.outputs().length);
        assertEquals(3, context.preActivations().length);
    }
    
    @Test
    void testWeightInitializationNotZero() {
        DenseLayer layer = new DenseLayer(mockOptimizer, mockActivator, 10, 5, WeightInitStrategy.HE);
        float[] input = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // Non-zero input
        
        Layer.LayerContext context = layer.forward(input, false);
        
        // Check that weights were initialized (not all zeros)
        boolean hasNonZero = false;
        for (float value : context.preActivations()) {
            if (value != 0.0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Weights should be initialized to non-zero values");
    }
    
    @Test
    void testBackwardPropagation() {
        DenseLayer layer = new DenseLayer(mockOptimizer, mockActivator, 3, 2, WeightInitStrategy.XAVIER);
        float[] input = {1.0f, 2.0f};
        
        // Forward pass
        Layer.LayerContext context = layer.forward(input, false);
        Layer.LayerContext[] stack = {context};
        
        // Backward pass
        float[] upstreamGradient = {0.1f, 0.2f, 0.3f};
        float[] downstreamGradient = layer.backward(stack, 0, upstreamGradient);
        
        assertNotNull(downstreamGradient);
        assertEquals(2, downstreamGradient.length); // Should match input size
    }
    
    @Test
    void testLayerSpec() {
        Layer.Spec spec = DenseLayer.spec(5, mockActivator, mockOptimizer, WeightInitStrategy.HE);
        
        assertEquals(5, spec.getOutputSize());
        
        Layer layer = spec.create(10);
        assertInstanceOf(DenseLayer.class, layer);
        assertEquals(5, layer.getOutputSize());
    }
    
    @Test
    void testWeightInitStrategies() {
        // Test that different strategies produce different initializations
        DenseLayer heLayer = new DenseLayer(mockOptimizer, mockActivator, 5, 4, WeightInitStrategy.HE);
        DenseLayer xavierLayer = new DenseLayer(mockOptimizer, mockActivator, 5, 4, WeightInitStrategy.XAVIER);
        
        float[] input = {1.0f, 1.0f, 1.0f, 1.0f};
        
        Layer.LayerContext heContext = heLayer.forward(input, false);
        Layer.LayerContext xavierContext = xavierLayer.forward(input, false);
        
        // Both should produce non-zero outputs
        boolean heHasNonZero = false;
        boolean xavierHasNonZero = false;
        
        for (float value : heContext.preActivations()) {
            if (value != 0.0f) heHasNonZero = true;
        }
        
        for (float value : xavierContext.preActivations()) {
            if (value != 0.0f) xavierHasNonZero = true;
        }
        
        assertTrue(heHasNonZero, "He initialization should produce non-zero values");
        assertTrue(xavierHasNonZero, "Xavier initialization should produce non-zero values");
        
        // The outputs should generally be different (very unlikely to be identical)
        assertFalse(java.util.Arrays.equals(heContext.preActivations(), xavierContext.preActivations()),
                   "Different initialization strategies should produce different results");
    }
    
    @Test
    void testLayerSpecWithCustomInit() {
        Layer.Spec spec = DenseLayer.spec(5, mockActivator, mockOptimizer, WeightInitStrategy.XAVIER);
        
        assertEquals(5, spec.getOutputSize());
        
        Layer layer = spec.create(10);
        assertInstanceOf(DenseLayer.class, layer);
        assertEquals(5, layer.getOutputSize());
    }
}