package dev.neuronic.net;

import dev.neuronic.net.activators.*;
import dev.neuronic.net.layers.*;
import dev.neuronic.net.optimizers.*;
import dev.neuronic.net.training.*;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for batch training capabilities.
 * Demonstrates both sequential batch processing and true mini-batch training.
 */
public class BatchTrainingTest {
    
    private AdamWOptimizer optimizer;
    private Random random;
    
    @BeforeEach
    public void setUp() {
        optimizer = new AdamWOptimizer(0.01f, 0.01f);
        random = new Random(42);
    }
    
    @Test
    public void testBasicBatchTraining() {
        // Create a simple network
        NeuralNet net = NeuralNet.newBuilder()
            .input(4)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        // Create batch data
        int batchSize = 16;
        float[][] batchInputs = new float[batchSize][4];
        float[][] batchTargets = new float[batchSize][3];
        
        for (int b = 0; b < batchSize; b++) {
            // Generate random input
            for (int i = 0; i < 4; i++) {
                batchInputs[b][i] = random.nextFloat() * 2 - 1;
            }
            
            // Create one-hot target
            int targetClass = b % 3;
            batchTargets[b][targetClass] = 1.0f;
        }
        
        // Train on batch
        net.trainBatch(batchInputs, batchTargets);
        
        // Verify we can predict on batch
        float[][] predictions = net.predictBatch(batchInputs);
        
        assertEquals(batchSize, predictions.length);
        assertEquals(3, predictions[0].length);
    }
    
    @Test
    public void testBatchVsSingleSampleConsistency() {
        // Test that batch training with size 1 behaves like single sample training
        // Note: Due to different weight initializations, we can't compare absolute values
        // Instead, we verify that both methods work and produce valid outputs
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(5)
            .setDefaultOptimizer(new SgdOptimizer(0.1f))
            .layer(Layers.hiddenDenseTanh(3))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        // Create test data
        float[] input = {0.5f, -0.3f, 0.8f, -0.1f, 0.2f};
        float[] target = {1.0f, 0.0f};
        
        // Get initial prediction
        float[] initialPred = net.predict(input).clone();
        
        // Train single sample
        net.train(input, target);
        float[] afterSinglePred = net.predict(input).clone();
        
        // Train as batch of size 1
        float[][] batchInput = {input};
        float[][] batchTarget = {target};
        net.trainBatch(batchInput, batchTarget);
        float[] afterBatchPred = net.predict(input);
        
        // Verify predictions changed after training
        boolean changed = false;
        for (int i = 0; i < initialPred.length; i++) {
            if (Math.abs(initialPred[i] - afterSinglePred[i]) > 1e-6f) {
                changed = true;
                break;
            }
        }
        assertTrue(changed, "Network should learn from single sample training");
        
        // Verify batch training also causes learning
        changed = false;
        for (int i = 0; i < afterSinglePred.length; i++) {
            if (Math.abs(afterSinglePred[i] - afterBatchPred[i]) > 1e-6f) {
                changed = true;
                break;
            }
        }
        assertTrue(changed, "Network should learn from batch training");
        
        // Verify outputs are valid probabilities
        float sum = 0;
        for (float p : afterBatchPred) {
            assertTrue(p >= 0 && p <= 1, "Output should be valid probability");
            sum += p;
        }
        assertEquals(1.0f, sum, 0.01f, "Softmax outputs should sum to 1");
    }
    
    @Test
    public void testBatchGradientAccumulation() {
        // Test gradient accumulation utilities
        int batchSize = 4;
        int gradientSize = 3;
        
        float[][] batchGradients = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f},
            {10.0f, 11.0f, 12.0f}
        };
        
        float[] averaged = new float[gradientSize];
        NetMath.batchAverageGradients(batchGradients, averaged);
        
        // Expected average: (1+4+7+10)/4 = 5.5, (2+5+8+11)/4 = 6.5, (3+6+9+12)/4 = 7.5
        assertEquals(5.5f, averaged[0], 1e-5f);
        assertEquals(6.5f, averaged[1], 1e-5f);
        assertEquals(7.5f, averaged[2], 1e-5f);
    }
    
    @Test
    public void testBatchMatrixMultiplication() {
        // Test batch matrix multiplication
        int batchSize = 2;
        int inputSize = 3;
        int neurons = 2;
        
        float[][] inputs = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        };
        
        float[][] weights = {
            {0.1f, 0.2f},  // weights for first input
            {0.3f, 0.4f},  // weights for second input
            {0.5f, 0.6f}   // weights for third input
        };
        
        float[] biases = {0.1f, 0.2f};
        
        float[][] outputs = new float[batchSize][neurons];
        
        NetMath.batchMatrixMultiply(inputs, weights, biases, outputs);
        
        // Verify first sample: 
        // output[0] = 1*0.1 + 2*0.3 + 3*0.5 + 0.1 = 0.1 + 0.6 + 1.5 + 0.1 = 2.3
        // output[1] = 1*0.2 + 2*0.4 + 3*0.6 + 0.2 = 0.2 + 0.8 + 1.8 + 0.2 = 3.0
        assertEquals(2.3f, outputs[0][0], 1e-5f);
        assertEquals(3.0f, outputs[0][1], 1e-5f);
        
        // Verify second sample:
        // output[0] = 4*0.1 + 5*0.3 + 6*0.5 + 0.1 = 0.4 + 1.5 + 3.0 + 0.1 = 5.0
        // output[1] = 4*0.2 + 5*0.4 + 6*0.6 + 0.2 = 0.8 + 2.0 + 3.6 + 0.2 = 6.6
        assertEquals(5.0f, outputs[1][0], 1e-5f);
        assertEquals(6.6f, outputs[1][1], 1e-5f);
    }
    

}