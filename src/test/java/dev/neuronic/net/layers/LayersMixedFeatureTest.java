package dev.neuronic.net.layers;

// LayerContext accessed as Layer.LayerContext
import dev.neuronic.net.Layers;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Layers factory methods for mixed feature input layers.
 */
class LayersMixedFeatureTest {

    private AdamWOptimizer optimizer;

    @BeforeEach
    void setUp() {
        optimizer = new AdamWOptimizer(0.001f, 0.01f);
    }

    @Test
    void testInputMixedBasic() {
        Layer.Spec spec = Layers.inputMixed(optimizer,
            Feature.embedding(1000, 16),
            Feature.oneHot(4),
            Feature.passthrough()
        );
        
        Layer layer = spec.create(0); // Input size ignored for input layers
        assertEquals(21, layer.getOutputSize()); // 16 + 4 + 1 = 21
        
        assertTrue(layer instanceof MixedFeatureInputLayer);
        MixedFeatureInputLayer mixedLayer = (MixedFeatureInputLayer) layer;
        
        Feature[] features = mixedLayer.getFeatures();
        assertEquals(3, features.length);
        assertEquals(Feature.Type.EMBEDDING, features[0].getType());
        assertEquals(Feature.Type.ONEHOT, features[1].getType());
        assertEquals(Feature.Type.PASSTHROUGH, features[2].getType());
    }

    @Test
    void testInputMixedWithDefaultOptimizer() {
        // Test that layer creation works when default optimizer will be provided by network builder
        Layer.Spec spec = Layers.inputMixed(
            Feature.embedding(500, 8),
            Feature.oneHot(3)
        );
        
        // Create layer with an optimizer (simulating what NeuralNetBuilder would do)
        Layer layer = ((MixedFeatureInputLayer.MixedFeatureInputLayerSpec) spec).createLayer(0, optimizer);
        assertEquals(11, layer.getOutputSize()); // 8 + 3 = 11
        assertTrue(layer instanceof MixedFeatureInputLayer);
    }

    @Test
    void testInputMixedWithInitStrategy() {
        Layer.Spec spec = Layers.inputMixed(optimizer, WeightInitStrategy.HE,
            Feature.embedding(100, 4),
            Feature.passthrough()
        );
        
        Layer layer = spec.create(0);
        assertEquals(5, layer.getOutputSize()); // 4 + 1 = 5
        assertTrue(layer instanceof MixedFeatureInputLayer);
    }

    @Test
    void testInputAllEmbeddings() {
        Layer.Spec spec = Layers.inputAllEmbeddings(32, optimizer, 
            10000, 5000, 2000); // Three features with different vocab sizes, same embedding dim
        
        Layer layer = spec.create(0);
        assertEquals(96, layer.getOutputSize()); // 3 * 32 = 96
        
        assertTrue(layer instanceof MixedFeatureInputLayer);
        MixedFeatureInputLayer mixedLayer = (MixedFeatureInputLayer) layer;
        
        Feature[] features = mixedLayer.getFeatures();
        assertEquals(3, features.length);
        
        // All should be embedding features
        for (Feature feature : features) {
            assertEquals(Feature.Type.EMBEDDING, feature.getType());
            assertEquals(32, feature.getEmbeddingDimension());
        }
        
        // Check different vocab sizes
        assertEquals(10000, features[0].getMaxUniqueValues());
        assertEquals(5000, features[1].getMaxUniqueValues());
        assertEquals(2000, features[2].getMaxUniqueValues());
    }

    @Test
    void testInputAllOneHot() {
        Layer.Spec spec = Layers.inputAllOneHot(optimizer, 4, 8, 3, 7);
        
        Layer layer = spec.create(0);
        assertEquals(22, layer.getOutputSize()); // 4 + 8 + 3 + 7 = 22
        
        assertTrue(layer instanceof MixedFeatureInputLayer);
        MixedFeatureInputLayer mixedLayer = (MixedFeatureInputLayer) layer;
        
        Feature[] features = mixedLayer.getFeatures();
        assertEquals(4, features.length);
        
        // All should be one-hot features
        for (Feature feature : features) {
            assertEquals(Feature.Type.ONEHOT, feature.getType());
        }
        
        // Check category counts
        assertEquals(4, features[0].getMaxUniqueValues());
        assertEquals(8, features[1].getMaxUniqueValues());
        assertEquals(3, features[2].getMaxUniqueValues());
        assertEquals(7, features[3].getMaxUniqueValues());
    }

    @Test
    void testInputAllNumerical() {
        // Note: numerical features don't need optimizers since they have no learnable parameters
        // But we still need one for API consistency
        Layer.Spec spec = Layers.inputAllNumerical(5);
        
        // Create layer with dummy optimizer (won't be used since no learnable parameters)
        Layer layer = ((MixedFeatureInputLayer.MixedFeatureInputLayerSpec) spec).createLayer(0, optimizer);
        assertEquals(5, layer.getOutputSize());
        
        assertTrue(layer instanceof MixedFeatureInputLayer);
        MixedFeatureInputLayer mixedLayer = (MixedFeatureInputLayer) layer;
        
        Feature[] features = mixedLayer.getFeatures();
        assertEquals(5, features.length);
        
        // All should be passthrough features
        for (Feature feature : features) {
            assertEquals(Feature.Type.PASSTHROUGH, feature.getType());
        }
    }

    @Test
    void testAdvertisingWorkflow() {
        // Create an advertising-style mixed feature layer
        Layer.Spec inputSpec = Layers.inputMixed(optimizer,
            Feature.embedding(100000, 64),   // bundle_id
            Feature.embedding(50000, 32),    // publisher_id
            Feature.oneHot(4),               // connection_type
            Feature.oneHot(8),               // device_type
            Feature.passthrough()            // user_age
        );
        
        Layer inputLayer = inputSpec.create(0);
        assertEquals(109, inputLayer.getOutputSize()); // 64 + 32 + 4 + 8 + 1 = 109
        
        // Test with realistic advertising data
        float[] adFeatures = {12345, 6789, 2, 5, 28.5f};
        Layer.LayerContext context = inputLayer.forward(adFeatures, false);
        
        float[] output = context.outputs();
        assertEquals(109, output.length);
        
        // Verify one-hot encoding for connection_type (position 96-99)
        // Input was 2, so output[98] should be 1.0, others 0.0
        assertEquals(0.0f, output[96]);  // category 0
        assertEquals(0.0f, output[97]);  // category 1  
        assertEquals(1.0f, output[98]);  // category 2 (active)
        assertEquals(0.0f, output[99]);  // category 3
        
        // Verify passthrough for user_age (position 108)
        assertEquals(28.5f, output[108], 1e-6f);
    }

    @Test
    void testConvenienceMethodsEquivalence() {
        // Test that convenience methods produce equivalent results to manual configuration
        
        // All embeddings: manual vs convenience
        Feature[] manualEmbeddings = {
            Feature.embedding(1000, 16),
            Feature.embedding(2000, 16),
            Feature.embedding(500, 16)
        };
        Layer.Spec manualSpec = Layers.inputMixed(optimizer, manualEmbeddings);
        Layer.Spec convenienceSpec = Layers.inputAllEmbeddings(16, optimizer, 1000, 2000, 500);
        
        Layer manualLayer = manualSpec.create(0);
        Layer convenienceLayer = convenienceSpec.create(0);
        
        assertEquals(manualLayer.getOutputSize(), convenienceLayer.getOutputSize());
        assertEquals(48, manualLayer.getOutputSize()); // 3 * 16 = 48
        
        // All one-hot: manual vs convenience
        Feature[] manualOneHot = {
            Feature.oneHot(3),
            Feature.oneHot(5),
            Feature.oneHot(2)
        };
        Layer.Spec manualOneHotSpec = Layers.inputMixed(optimizer, manualOneHot);
        Layer.Spec convenienceOneHotSpec = Layers.inputAllOneHot(optimizer, 3, 5, 2);
        
        Layer manualOneHotLayer = manualOneHotSpec.create(0);
        Layer convenienceOneHotLayer = convenienceOneHotSpec.create(0);
        
        assertEquals(manualOneHotLayer.getOutputSize(), convenienceOneHotLayer.getOutputSize());
        assertEquals(10, manualOneHotLayer.getOutputSize()); // 3 + 5 + 2 = 10
        
        // All numerical: manual vs convenience
        Feature[] manualNumerical = {
            Feature.passthrough(),
            Feature.passthrough(),
            Feature.passthrough(),
            Feature.passthrough()
        };
        Layer.Spec manualNumericalSpec = Layers.inputMixed(optimizer, manualNumerical);
        Layer.Spec convenienceNumericalSpec = Layers.inputAllNumerical(4);
        
        Layer manualNumericalLayer = manualNumericalSpec.create(0);
        // Create convenience layer with optimizer (since numerical features still need optimizer API consistency)
        Layer convenienceNumericalLayer = ((MixedFeatureInputLayer.MixedFeatureInputLayerSpec) convenienceNumericalSpec).createLayer(0, optimizer);
        
        assertEquals(manualNumericalLayer.getOutputSize(), convenienceNumericalLayer.getOutputSize());
        assertEquals(4, manualNumericalLayer.getOutputSize());
    }

    @Test
    void testInputValidationInFactoryMethods() {
        // Test empty arrays
        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllEmbeddings(32, optimizer);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllOneHot(optimizer);
        });
        
        // Test invalid embedding dimension
        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllEmbeddings(0, optimizer, 1000);
        });
        
        // Test invalid vocab sizes
        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllEmbeddings(32, optimizer, 0);
        });
        
        // Test invalid category counts
        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllOneHot(optimizer, 0);
        });
        
        // Test invalid number of numerical features
        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllNumerical(0);
        });
    }

    @Test
    void testMixedFeatureForwardBackwardIntegration() {
        // Test that mixed feature layers work correctly in forward/backward passes
        Layer.Spec spec = Layers.inputMixed(optimizer,
            Feature.embedding(10, 4),
            Feature.oneHot(3),
            Feature.passthrough()
        );
        
        Layer layer = spec.create(0);
        
        // Forward pass
        float[] input = {5, 1, 2.5f};
        Layer.LayerContext context = layer.forward(input, false);
        
        assertEquals(8, context.outputs().length); // 4 + 3 + 1 = 8
        
        // Verify one-hot encoding
        float[] output = context.outputs();
        assertEquals(0.0f, output[4]); // category 0
        assertEquals(1.0f, output[5]); // category 1 (active)
        assertEquals(0.0f, output[6]); // category 2
        
        // Verify passthrough
        assertEquals(2.5f, output[7], 1e-6f);
        
        // Backward pass (should not throw)
        float[] gradient = new float[8];
        for (int i = 0; i < 8; i++) {
            gradient[i] = 0.1f;
        }
        
        Layer.LayerContext[] stack = {context};
        assertDoesNotThrow(() -> {
            layer.backward(stack, 0, gradient);
        });
    }
}