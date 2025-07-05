
package dev.neuronic.net.optimizers.adamw;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class FusedAdamWUpdateVectorTest {

    @Test
    public void testAgainstGoldenValues() {
        // Inputs
        float[] params = {0.5f, -0.2f};
        float[] gradients = {0.1f, 0.3f};
        float[] momentum = {0.0f, 0.0f};
        float[] velocity = {0.0f, 0.0f};
        float learningRate = 0.001f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        float weightDecay = 0.01f;
        long timeStep = 1;

        // Expected values (calculated manually based on PyTorch's implementation)
        float[] expectedParams = {0.49401f, -0.19899f};

        // Run the vectorized implementation
        FusedAdamWUpdate.computeVectorized(params, gradients, momentum, velocity, beta1, beta2, learningRate, epsilon, weightDecay, 1.0f - beta1, 1.0f - beta2, true);

        // Assert
        assertArrayEquals(expectedParams, params, 1e-6f);
    }
}
