package dev.neuronic;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class XorTest {
    
    @Test
    public void testXORLearning() {
        // Create simple XOR network with appropriate layers
        NeuralNet net = NeuralNet.newBuilder()
                .input(2)
                .setDefaultOptimizer(new AdamWOptimizer(0.015f, 0.0f))  // AdamW for stable learning
                .layer(Layers.hiddenDenseRelu(8))  // Dense layer with ReLU - appropriate for XOR
                .output(Layers.outputSigmoidBinary());
        
        float[][] inputs = {
            {0.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f}
        };
        float[][] targets = {
            {0.0f}, {1.0f}, {1.0f}, {0.0f}
        };
        
        // Train for sufficient epochs
        for (int epoch = 0; epoch < 500; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                net.train(inputs[i], targets[i]);
            }
        }
        
        // Check final predictions
        float totalError = 0;
        for (int i = 0; i < inputs.length; i++) {
            float[] pred = net.predict(inputs[i]);
            float error = Math.abs(pred[0] - targets[i][0]);
            totalError += error;
        }
        float avgError = totalError / 4;
        
        // XOR should achieve < 0.2 average error with proper architecture
        assertTrue(avgError < 0.2f, 
            "XOR should learn with average error < 0.2, but got " + avgError);
    }

}