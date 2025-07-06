package dev.neuronic.net.layers;

import dev.neuronic.net.Layers;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for comprehensive validation in mixed feature input layers.
 * Verifies that user errors are caught early with helpful error messages.
 */
class MixedFeatureValidationTest {

    private AdamWOptimizer optimizer;

    @BeforeEach
    void setUp() {
        optimizer = new AdamWOptimizer(0.001f, 0.01f);
    }

    // Feature configuration validation tests

    @Test
    void testFeatureConfigurationValidation() {
        // Test null features array
        assertThrows(IllegalArgumentException.class, () -> {
            new MixedFeatureInputLayer(optimizer, null, WeightInitStrategy.XAVIER);
        });

        // Test empty features array
        assertThrows(IllegalArgumentException.class, () -> {
            new MixedFeatureInputLayer(optimizer, new Feature[0], WeightInitStrategy.XAVIER);
        });

        // Test null feature in array
        Feature[] featuresWithNull = {Feature.embedding(100, 16), null, Feature.oneHot(4)};
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            new MixedFeatureInputLayer(optimizer, featuresWithNull, WeightInitStrategy.XAVIER);
        });
        assertTrue(exception.getMessage().contains("Feature 1 is null"));
    }

    // Factory method validation tests

    @Test
    void testInputMixedValidation() {
        // Test null feature configurations
        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputMixed(optimizer, (Feature[]) null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputMixed((Feature[]) null);
        });

        // Test empty feature configurations
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputMixed(optimizer);
        });
        assertTrue(exception.getMessage().contains("At least one feature must be configured"));
        assertTrue(exception.getMessage().contains("Example"));
    }

    @Test
    void testInputAllEmbeddingsValidation() {
        // Test invalid embedding dimension
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllEmbeddings(0, optimizer, 1000);
        });
        assertTrue(exception.getMessage().contains("Embedding dimension must be positive"));

        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllEmbeddings(-1, optimizer, 1000);
        });

        // Test null vocab sizes
        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllEmbeddings(32, optimizer, (int[]) null);
        });

        // Test empty vocab sizes
        exception = assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllEmbeddings(32, optimizer);
        });
        assertTrue(exception.getMessage().contains("At least one feature must be configured"));

        // Test invalid vocab size
        exception = assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllEmbeddings(32, optimizer, 1000, 0, 500);
        });
        assertTrue(exception.getMessage().contains("Feature 1: maxUniqueValues must be positive"));
    }

    @Test
    void testInputAllOneHotValidation() {
        // Test null category sizes
        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllOneHot((int[]) null);
        });

        // Test empty category sizes
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllOneHot();
        });
        assertTrue(exception.getMessage().contains("At least one feature must be configured"));

        // Test invalid category size
        exception = assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllOneHot(4, -1, 8);
        });
        assertTrue(exception.getMessage().contains("Feature 1: numberOfCategories must be positive"));
    }

    @Test
    void testInputAllNumericalValidation() {
        // Test invalid number of features
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllNumerical(0);
        });
        assertTrue(exception.getMessage().contains("Number of features must be positive"));
        assertTrue(exception.getMessage().contains("Example"));

        assertThrows(IllegalArgumentException.class, () -> {
            Layers.inputAllNumerical(-1);
        });
    }

    // Runtime input validation tests

    @Test
    void testForwardInputValidation() {
        Feature[] features = {
            Feature.embedding(100, 8),
            Feature.oneHot(4),
            Feature.passthrough()
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);

        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(null, false);
        });

        // Test wrong input size with helpful error message
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{1.0f, 2.0f}, false); // Too few inputs
        });
        assertTrue(exception.getMessage().contains("Input array has 2 elements but 3 features were configured"));
        assertTrue(exception.getMessage().contains("Expected input format"));

        exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, false); // Too many inputs
        });
        assertTrue(exception.getMessage().contains("Input array has 4 elements but 3 features were configured"));
    }

    @Test
    void testEmbeddingValueValidation() {
        Feature[] features = {Feature.embedding(10, 4)};
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);

        // Test negative embedding value
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{-1.0f}, false);
        });
        assertTrue(exception.getMessage().contains("Feature 0 (embedding): value -1 is out of range"));
        assertTrue(exception.getMessage().contains("Embedding features expect token/category IDs"));

        // Test embedding value too large
        exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{10.0f}, false); // vocab size is 10, so valid range is 0-9
        });
        assertTrue(exception.getMessage().contains("Feature 0 (embedding): value 10 is out of range [0, 10)"));
        assertTrue(exception.getMessage().contains("increase maxUniqueValues to 11"));

        // Test non-integer embedding value
        exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{5.5f}, false);
        });
        assertTrue(exception.getMessage().contains("Feature 0 (EMBEDDING): input must be integer, got 5.50"));
        assertTrue(exception.getMessage().contains("Use Feature.passthrough(), Feature.autoScale(minBound, maxBound), or Feature.autoNormalize() for continuous numerical values"));
    }

    @Test
    void testOneHotValueValidation() {
        Feature[] features = {Feature.oneHot(3)};
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);

        // Test negative one-hot value
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{-1.0f}, false);
        });
        assertTrue(exception.getMessage().contains("Feature 0 (oneHot): value -1 is out of range"));

        // Test one-hot value too large
        exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{3.0f}, false); // 3 categories, so valid range is 0-2
        });
        assertTrue(exception.getMessage().contains("Feature 0 (oneHot): value 3 is out of range [0, 3)"));
        assertTrue(exception.getMessage().contains("increase numberOfCategories to 4"));

        // Test non-integer one-hot value
        exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{1.5f}, false);
        });
        assertTrue(exception.getMessage().contains("Feature 0 (ONEHOT): input must be integer, got 1.50"));
    }

    @Test
    void testPassthroughValueValidation() {
        Feature[] features = {Feature.passthrough()};
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);

        // Passthrough should accept any float value
        assertDoesNotThrow(() -> {
            layer.forward(new float[]{-123.456f}, false);
        });

        assertDoesNotThrow(() -> {
            layer.forward(new float[]{Float.MAX_VALUE}, false);
        });

        assertDoesNotThrow(() -> {
            layer.forward(new float[]{0.0f}, false);
        });
    }

    @Test
    void testMixedFeatureErrorMessages() {
        Feature[] features = {
            Feature.embedding(100, 16),  // Feature 0
            Feature.oneHot(4),           // Feature 1
            Feature.passthrough()        // Feature 2
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);

        // Test embedding error includes feature index
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{150.0f, 2.0f, 3.14f}, false);
        });
        assertTrue(exception.getMessage().contains("Feature 0 (embedding)"));
        assertTrue(exception.getMessage().contains("value 150 is out of range"));

        // Test one-hot error includes feature index
        exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{50.0f, 5.0f, 3.14f}, false);
        });
        assertTrue(exception.getMessage().contains("Feature 1 (oneHot)"));
        assertTrue(exception.getMessage().contains("value 5 is out of range"));

        // Test non-integer error includes feature index and type
        exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{50.5f, 2.0f, 3.14f}, false);
        });
        assertTrue(exception.getMessage().contains("Feature 0 (EMBEDDING)"));
        assertTrue(exception.getMessage().contains("input must be integer, got 50.50"));
    }

    @Test
    void testValidInputsWork() {
        // Verify that valid inputs still work correctly after adding validation
        Feature[] features = {
            Feature.embedding(100, 8),
            Feature.oneHot(4),
            Feature.passthrough()
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.XAVIER);

        // Valid input should work without exceptions
        assertDoesNotThrow(() -> {
            Layer.LayerContext context = layer.forward(new float[]{42.0f, 2.0f, 3.14159f}, false);
            assertEquals(13, context.outputs().length); // 8 + 4 + 1 = 13
        });
    }
}