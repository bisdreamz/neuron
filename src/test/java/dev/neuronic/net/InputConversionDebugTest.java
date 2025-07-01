package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

public class InputConversionDebugTest {

    @Test
    public void testPremiumVsRegularInputConversion() throws Exception {
        System.out.println("=== Input Conversion Debug Test ===\n");

        // 1. Configure the model exactly like in the failing test
        Feature[] features = {
            Feature.oneHot(50, "OS"),
            Feature.embeddingLRU(20_000, 32, "ZONEID"),
            Feature.embeddingLRU(20_000, 32, "DOMAIN"),
            Feature.embeddingLRU(20_000, 32, "PUB"),
            Feature.autoScale(0f, 20f, "BIDFLOOR")
        };

        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(new dev.neuronic.net.optimizers.AdamWOptimizer(0.001f, 0.01f))
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(8)) // Smaller for a quick test
                .output(Layers.outputHuberRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);

        // 2. Create one premium and one regular input map
        // Premium Segment: ZONEID and DOMAIN are small integers
        Map<String, Object> premiumInput = Map.of(
            "OS", 1,
            "ZONEID", 10,
            "DOMAIN", 10,
            "PUB", 123,
            "BIDFLOOR", 2.5f
        );

        // Regular Segment: ZONEID and DOMAIN are larger integers
        Map<String, Object> regularInput = Map.of(
            "OS", 1,
            "ZONEID", 15000,
            "DOMAIN", 4500,
            "PUB", 456,
            "BIDFLOOR", 0.8f
        );
        
        // 3. Use reflection to access the protected convertFromMap method
        Method convertMethod = SimpleNet.class.getDeclaredMethod("convertFromMap", Map.class);
        convertMethod.setAccessible(true);

        // 4. Convert both inputs to float arrays
        float[] premiumFloatArray = (float[]) convertMethod.invoke(model, premiumInput);
        float[] regularFloatArray = (float[]) convertMethod.invoke(model, regularInput);

        // 5. Print the arrays for visual inspection
        // System.out.println("Feature Names: " + Arrays.toString(model.getFeatureNames()));
        System.out.println("Premium Input Map: " + premiumInput);
        System.out.println("Premium Float Array: " + Arrays.toString(premiumFloatArray) + "\n");
        
        System.out.println("Regular Input Map: " + regularInput);
        System.out.println("Regular Float Array: " + Arrays.toString(regularFloatArray) + "\n");

        // 6. Assert that the arrays are not identical
        System.out.println("Are the generated float arrays identical? " + Arrays.equals(premiumFloatArray, regularFloatArray));
        assertFalse(Arrays.equals(premiumFloatArray, regularFloatArray), "The float arrays for premium and regular inputs should not be identical.");
        System.out.println("\n[SUCCESS] The conversion logic correctly produces different arrays for different inputs.");
    }
}