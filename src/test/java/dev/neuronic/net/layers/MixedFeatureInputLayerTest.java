package dev.neuronic.net.layers;

// LayerContext accessed as Layer.LayerContext
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for mixed feature input layer functionality.
 */
class MixedFeatureInputLayerTest {

    private SgdOptimizer optimizer;
    private AdamWOptimizer adamOptimizer;

    @BeforeEach
    void setUp() {
        optimizer = new SgdOptimizer(0.01f);
        adamOptimizer = new AdamWOptimizer(0.001f, 0.01f);
    }

    @Test
    void testFeatureConfiguration() {
        // Test Feature factory methods
        Feature embedding = Feature.embedding(1000, 32);
        Feature oneHot = Feature.oneHot(4);
        Feature passthrough = Feature.passthrough();
        
        assertEquals(Feature.Type.EMBEDDING, embedding.getType());
        assertEquals(1000, embedding.getMaxUniqueValues());
        assertEquals(32, embedding.getEmbeddingDimension());
        assertEquals(32, embedding.getOutputDimension());
        
        assertEquals(Feature.Type.ONEHOT, oneHot.getType());
        assertEquals(4, oneHot.getMaxUniqueValues());
        assertEquals(4, oneHot.getOutputDimension());
        
        assertEquals(Feature.Type.PASSTHROUGH, passthrough.getType());
        assertEquals(1, passthrough.getOutputDimension());
    }
    
    @Test
    void testFeatureValidation() {
        // Test invalid parameters
        assertThrows(IllegalArgumentException.class, () -> Feature.embedding(0, 32));
        assertThrows(IllegalArgumentException.class, () -> Feature.embedding(1000, 0));
        assertThrows(IllegalArgumentException.class, () -> Feature.embedding(-1, 32));
        assertThrows(IllegalArgumentException.class, () -> Feature.oneHot(0));
        assertThrows(IllegalArgumentException.class, () -> Feature.oneHot(-1));
    }

    @Test
    void testMixedFeatureLayerBasic() {
        Feature[] features = {
            Feature.embedding(100, 16),  // input[0]: embedding feature
            Feature.oneHot(4),           // input[1]: one-hot feature
            Feature.passthrough()        // input[2]: numerical feature
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        
        // Test output size calculation: 16 + 4 + 1 = 21
        assertEquals(21, layer.getOutputSize());
        
        // Test feature access
        Feature[] layerFeatures = layer.getFeatures();
        assertEquals(3, layerFeatures.length);
        assertEquals(Feature.Type.EMBEDDING, layerFeatures[0].getType());
        assertEquals(Feature.Type.ONEHOT, layerFeatures[1].getType());
        assertEquals(Feature.Type.PASSTHROUGH, layerFeatures[2].getType());
    }

    @Test
    void testForwardPassMixed() {
        Feature[] features = {
            Feature.embedding(1000, 8),   // input[0]: embedding (vocab=1000, dim=8)
            Feature.oneHot(3),            // input[1]: one-hot (3 categories)
            Feature.passthrough()         // input[2]: numerical
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        
        // Test forward pass
        float[] input = {42.0f, 1.0f, 3.14f}; // embedding_id=42, category=1, value=3.14
        Layer.LayerContext context = layer.forward(input);
        
        float[] output = context.outputs();
        assertEquals(12, output.length); // 8 + 3 + 1 = 12
        
        // Check embedding output (first 8 elements should be non-zero from random initialization)
        boolean hasNonZeroEmbedding = false;
        for (int i = 0; i < 8; i++) {
            if (output[i] != 0.0f) {
                hasNonZeroEmbedding = true;
                break;
            }
        }
        assertTrue(hasNonZeroEmbedding, "Embedding should have non-zero values");
        
        // Check one-hot output (elements 8-10, with element 9 being 1.0)
        assertEquals(0.0f, output[8], 1e-6f);  // category 0
        assertEquals(1.0f, output[9], 1e-6f);  // category 1 (active)
        assertEquals(0.0f, output[10], 1e-6f); // category 2
        
        // Check passthrough output (element 11)
        assertEquals(3.14f, output[11], 1e-6f);
    }

    @Test
    void testAllEmbeddingFeatures() {
        Feature[] features = {
            Feature.embedding(100, 4),
            Feature.embedding(200, 4),
            Feature.embedding(50, 4)
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.HE);
        
        assertEquals(12, layer.getOutputSize()); // 3 * 4 = 12
        
        float[] input = {10.0f, 150.0f, 25.0f};
        Layer.LayerContext context = layer.forward(input);
        
        assertEquals(12, context.outputs().length);
        assertArrayEquals(input, context.inputs());
    }

    @Test
    void testAllOneHotFeatures() {
        Feature[] features = {
            Feature.oneHot(3),
            Feature.oneHot(5),
            Feature.oneHot(2)
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        
        assertEquals(10, layer.getOutputSize()); // 3 + 5 + 2 = 10
        
        float[] input = {1.0f, 3.0f, 0.0f}; // categories 1, 3, 0
        Layer.LayerContext context = layer.forward(input);
        
        float[] output = context.outputs();
        assertEquals(10, output.length);
        
        // First one-hot (3 categories): [0, 1, 0]
        assertEquals(0.0f, output[0]);
        assertEquals(1.0f, output[1]);
        assertEquals(0.0f, output[2]);
        
        // Second one-hot (5 categories): [0, 0, 0, 1, 0]
        assertEquals(0.0f, output[3]);
        assertEquals(0.0f, output[4]);
        assertEquals(0.0f, output[5]);
        assertEquals(1.0f, output[6]);
        assertEquals(0.0f, output[7]);
        
        // Third one-hot (2 categories): [1, 0]
        assertEquals(1.0f, output[8]);
        assertEquals(0.0f, output[9]);
    }

    @Test
    void testAllPassthroughFeatures() {
        Feature[] features = {
            Feature.passthrough(),
            Feature.passthrough(),
            Feature.passthrough()
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        
        assertEquals(3, layer.getOutputSize());
        
        float[] input = {1.5f, -2.3f, 42.0f};
        Layer.LayerContext context = layer.forward(input);
        
        assertArrayEquals(input, context.outputs(), 1e-6f);
    }

    @Test
    void testInputValidation() {
        Feature[] features = {
            Feature.embedding(100, 8),
            Feature.oneHot(4),
            Feature.passthrough()
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        
        // Test wrong input length
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{1.0f, 2.0f}); // Too few inputs
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{1.0f, 2.0f, 3.0f, 4.0f}); // Too many inputs
        });
        
        // Test out-of-range embedding values
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{-1.0f, 2.0f, 3.0f}); // Negative embedding ID
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{100.0f, 2.0f, 3.0f}); // Embedding ID >= vocab size
        });
        
        // Test out-of-range one-hot values
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{50.0f, -1.0f, 3.0f}); // Negative category
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{50.0f, 4.0f, 3.0f}); // Category >= num categories
        });
        
        // Test non-integer values for categorical features
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{50.5f, 2.0f, 3.0f}); // Non-integer embedding ID
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{50.0f, 2.5f, 3.0f}); // Non-integer category
        });
    }

    @Test
    void testBackwardPass() {
        Feature[] features = {
            Feature.embedding(10, 4),
            Feature.oneHot(3),
            Feature.passthrough()
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(adamOptimizer, features, WeightInitStrategy.XAVIER);
        
        // Forward pass
        float[] input = {5.0f, 1.0f, 2.5f};
        Layer.LayerContext context = layer.forward(input);
        
        // Store original embedding
        float[] originalEmbedding = layer.getEmbedding(0, 5);
        
        // Backward pass with gradient
        float[] upstreamGradient = new float[8]; // 4 + 3 + 1 = 8
        // Set gradient for embedding (first 4 elements)
        upstreamGradient[0] = 0.1f;
        upstreamGradient[1] = -0.2f;
        upstreamGradient[2] = 0.3f;
        upstreamGradient[3] = -0.1f;
        // One-hot gradients (elements 4-6) don't affect anything
        upstreamGradient[4] = 1.0f;
        upstreamGradient[5] = 2.0f;
        upstreamGradient[6] = 3.0f;
        // Passthrough gradient (element 7) doesn't affect anything
        upstreamGradient[7] = 0.5f;
        
        Layer.LayerContext[] stack = {context};
        layer.backward(stack, 0, upstreamGradient);
        
        // Check that embedding was updated
        float[] updatedEmbedding = layer.getEmbedding(0, 5);
        boolean wasUpdated = false;
        for (int i = 0; i < 4; i++) {
            if (originalEmbedding[i] != updatedEmbedding[i]) {
                wasUpdated = true;
                break;
            }
        }
        assertTrue(wasUpdated, "Embedding should be updated after backward pass");
    }

    @Test
    void testGetEmbedding() {
        Feature[] features = {
            Feature.embedding(5, 3),
            Feature.oneHot(2)
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        
        // Test valid embedding access
        float[] embedding = layer.getEmbedding(0, 2);
        assertEquals(3, embedding.length);
        
        // Test invalid feature index
        assertThrows(IllegalArgumentException.class, () -> {
            layer.getEmbedding(-1, 0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.getEmbedding(2, 0); // Only 2 features (indices 0, 1)
        });
        
        // Test accessing non-embedding feature
        assertThrows(IllegalArgumentException.class, () -> {
            layer.getEmbedding(1, 0); // Feature 1 is one-hot, not embedding
        });
        
        // Test invalid value index
        assertThrows(IllegalArgumentException.class, () -> {
            layer.getEmbedding(0, -1);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.getEmbedding(0, 5); // Vocab size is 5, so valid indices are 0-4
        });
    }

    @Test
    void testAdvertisingExample() {
        // Real-world advertising example
        Feature[] features = {
            Feature.embedding(100000, 64),  // bundle_id (100k bundles)
            Feature.embedding(50000, 32),   // publisher_id (50k publishers)
            Feature.oneHot(4),              // connection_type (wifi/4g/3g/other)
            Feature.oneHot(8),              // device_type (phone/tablet/etc)
            Feature.passthrough()           // user_age (continuous)
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(adamOptimizer, features, WeightInitStrategy.XAVIER);
        
        // Expected output dimension: 64 + 32 + 4 + 8 + 1 = 109
        assertEquals(109, layer.getOutputSize());
        
        // Test with advertising data
        float[] adInput = {
            12345.0f,  // bundle_id
            6789.0f,   // publisher_id
            2.0f,      // connection_type (3g)
            5.0f,      // device_type
            28.5f      // user_age
        };
        
        Layer.LayerContext context = layer.forward(adInput);
        float[] output = context.outputs();
        
        assertEquals(109, output.length);
        
        // Check connection_type one-hot (positions 96-99): should be [0, 0, 1, 0]
        assertEquals(0.0f, output[96]);  // wifi
        assertEquals(0.0f, output[97]);  // 4g
        assertEquals(1.0f, output[98]);  // 3g (active)
        assertEquals(0.0f, output[99]);  // other
        
        // Check user_age passthrough (position 108)
        assertEquals(28.5f, output[108], 1e-6f);
        
        // Check that embeddings have reasonable values (non-zero due to initialization)
        boolean hasNonZeroBundleEmbedding = false;
        for (int i = 0; i < 64; i++) {
            if (output[i] != 0.0f) {
                hasNonZeroBundleEmbedding = true;
                break;
            }
        }
        assertTrue(hasNonZeroBundleEmbedding, "Bundle embedding should have non-zero values");
    }

    @Test
    void testEmptyFeatureConfiguration() {
        assertThrows(IllegalArgumentException.class, () -> {
            new MixedFeatureInputLayer(optimizer, new Feature[0], WeightInitStrategy.XAVIER);
        });
    }

    @Test
    void testFeatureToString() {
        Feature embedding = Feature.embedding(1000, 32);
        Feature oneHot = Feature.oneHot(4);
        Feature passthrough = Feature.passthrough();
        
        assertTrue(embedding.toString().contains("embedding"));
        assertTrue(embedding.toString().contains("1000"));
        assertTrue(embedding.toString().contains("32"));
        
        assertTrue(oneHot.toString().contains("oneHot"));
        assertTrue(oneHot.toString().contains("4"));
        
        assertTrue(passthrough.toString().contains("passthrough"));
    }
    
    @Test
    void testFeatureNamesInLayer() {
        Feature[] features = {
            Feature.embedding(1000, 32, "user_id"),
            Feature.oneHot(4, "device_type"),
            Feature.passthrough("ctr"),
            Feature.autoScale(0.01f, 100.0f, "bid_floor"),
            Feature.autoNormalize("user_age")
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        
        String[] names = layer.getFeatureNames();
        assertEquals(5, names.length);
        assertEquals("user_id", names[0]);
        assertEquals("device_type", names[1]);
        assertEquals("ctr", names[2]);
        assertEquals("bid_floor", names[3]);
        assertEquals("user_age", names[4]);
        
        assertTrue(layer.hasExplicitFeatureNames());
    }
    
    @Test
    void testMixedExplicitAndImplicitNamesNotAllowed() {
        Feature[] features = {
            Feature.embedding(1000, 32, "user_id"),
            Feature.oneHot(4),  // Missing name
            Feature.passthrough("ctr")
        };
        
        // Should throw exception because of mixed naming
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> {
            new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        });
        
        assertTrue(ex.getMessage().contains("Feature naming must be all-or-nothing"));
        assertTrue(ex.getMessage().contains("2 named and 1 unnamed features"));
        assertTrue(ex.getMessage().contains("Feature.oneHot(4, \"your_feature_name\")"));
    }
    
    @Test
    void testAllUnnamedFeatures() {
        Feature[] features = {
            Feature.embedding(1000, 32),
            Feature.oneHot(4),
            Feature.passthrough()
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);
        
        String[] names = layer.getFeatureNames();
        assertEquals(3, names.length);
        assertNull(names[0]);
        assertNull(names[1]);
        assertNull(names[2]);
        
        assertFalse(layer.hasExplicitFeatureNames());
    }
    
    @Test
    void testFeatureCreationWithValidNames() {
        assertDoesNotThrow(() -> Feature.embedding(1000, 32, "valid_name"));
        assertDoesNotThrow(() -> Feature.oneHot(4, "deviceType"));
        assertDoesNotThrow(() -> Feature.passthrough("price"));
        assertDoesNotThrow(() -> Feature.autoScale(0.0f, 1.0f, "score"));
        assertDoesNotThrow(() -> Feature.autoNormalize("age"));
    }
    
    @Test
    void testFeatureCreationWithInvalidNames() {
        assertThrows(IllegalArgumentException.class, () -> Feature.embedding(1000, 32, null));
        assertThrows(IllegalArgumentException.class, () -> Feature.embedding(1000, 32, ""));
        assertThrows(IllegalArgumentException.class, () -> Feature.embedding(1000, 32, "   "));
        
        assertThrows(IllegalArgumentException.class, () -> Feature.oneHot(4, null));
        assertThrows(IllegalArgumentException.class, () -> Feature.oneHot(4, ""));
        
        assertThrows(IllegalArgumentException.class, () -> Feature.passthrough(null));
        assertThrows(IllegalArgumentException.class, () -> Feature.passthrough(""));
        
        assertThrows(IllegalArgumentException.class, () -> Feature.autoScale(0.0f, 1.0f, null));
        assertThrows(IllegalArgumentException.class, () -> Feature.autoScale(0.0f, 1.0f, ""));
        
        assertThrows(IllegalArgumentException.class, () -> Feature.autoNormalize(null));
        assertThrows(IllegalArgumentException.class, () -> Feature.autoNormalize(""));
    }
    
    @Test
    void testFeatureNameTrimming() {
        Feature f1 = Feature.embedding(100, 32, "  user_id  ");
        assertEquals("user_id", f1.getName());
        
        Feature f2 = Feature.oneHot(4, " device_type ");
        assertEquals("device_type", f2.getName());
        
        Feature f3 = Feature.passthrough(" ctr ");
        assertEquals("ctr", f3.getName());
        
        Feature f4 = Feature.autoScale(0.0f, 1.0f, " score ");
        assertEquals("score", f4.getName());
        
        Feature f5 = Feature.autoNormalize(" age ");
        assertEquals("age", f5.getName());
    }
    
    @Test
    void testFeatureToStringWithNames() {
        Feature f1 = Feature.embedding(1000, 32, "user_id");
        assertTrue(f1.toString().contains("[name=user_id]"));
        
        Feature f2 = Feature.embedding(1000, 32);
        assertFalse(f2.toString().contains("[name="));
        
        Feature f3 = Feature.oneHot(4, "device");
        assertTrue(f3.toString().contains("[name=device]"));
        
        Feature f4 = Feature.passthrough("price");
        assertTrue(f4.toString().contains("[name=price]"));
        
        Feature f5 = Feature.autoScale(0.0f, 100.0f, "score");
        assertTrue(f5.toString().contains("[name=score]"));
        
        Feature f6 = Feature.autoNormalize("age");
        assertTrue(f6.toString().contains("[name=age]"));
    }
}