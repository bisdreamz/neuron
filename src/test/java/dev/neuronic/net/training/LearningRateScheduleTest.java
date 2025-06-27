package dev.neuronic.net.training;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LearningRateScheduleTest {
    
    @Test
    public void testConstantSchedule() {
        LearningRateSchedule schedule = LearningRateSchedule.constant(0.001f);
        
        // Should remain constant throughout
        assertEquals(0.001f, schedule.getLearningRate(0, 100), 1e-6f);
        assertEquals(0.001f, schedule.getLearningRate(50, 100), 1e-6f);
        assertEquals(0.001f, schedule.getLearningRate(99, 100), 1e-6f);
    }
    
    @Test
    public void testStepDecay() {
        LearningRateSchedule schedule = LearningRateSchedule.stepDecay(0.1f, 0.1f, 3);
        
        // First 3 epochs: 0.1
        assertEquals(0.1f, schedule.getLearningRate(0, 10), 1e-6f);
        assertEquals(0.1f, schedule.getLearningRate(2, 10), 1e-6f);
        
        // Next 3 epochs: 0.01
        assertEquals(0.01f, schedule.getLearningRate(3, 10), 1e-6f);
        assertEquals(0.01f, schedule.getLearningRate(5, 10), 1e-6f);
        
        // Next 3 epochs: 0.001
        assertEquals(0.001f, schedule.getLearningRate(6, 10), 1e-6f);
    }
    
    @Test
    public void testCosineAnnealing() {
        LearningRateSchedule schedule = LearningRateSchedule.cosineAnnealing(0.001f, 10, 0);
        
        // Start at initial learning rate
        assertEquals(0.001f, schedule.getLearningRate(0, 10), 1e-6f);
        
        // Midpoint should be around 0.5 * initial
        float midpoint = schedule.getLearningRate(5, 10);
        assertTrue(midpoint < 0.001f);
        assertTrue(midpoint > 0.0001f);
        
        // End should be close to minimum (0.1% of initial)
        float end = schedule.getLearningRate(9, 10);
        assertTrue(end >= 0.001f * 0.001f); // Minimum threshold
        assertTrue(end < 0.0002f);
    }
    
    @Test
    public void testCosineAnnealingWithWarmup() {
        LearningRateSchedule schedule = LearningRateSchedule.cosineAnnealing(0.001f, 10, 3);
        
        // During warmup, should be constant
        assertEquals(0.001f, schedule.getLearningRate(0, 10), 1e-6f);
        assertEquals(0.001f, schedule.getLearningRate(2, 10), 1e-6f);
        
        // After warmup, should start decreasing
        float afterWarmup = schedule.getLearningRate(3, 10);
        assertEquals(0.001f, afterWarmup, 1e-6f); // First epoch after warmup still at initial
        
        float later = schedule.getLearningRate(6, 10);
        assertTrue(later < 0.001f);
    }
    
    @Test
    public void testPolynomialDecay() {
        LearningRateSchedule schedule = LearningRateSchedule.polynomialDecay(0.01f, 0.0001f, 10, 1.0f);
        
        // Linear decay (power=1.0)
        assertEquals(0.01f, schedule.getLearningRate(0, 10), 1e-6f);
        
        // Halfway should be halfway between initial and end
        float midpoint = schedule.getLearningRate(5, 10);
        float expected = 0.01f - (0.01f - 0.0001f) * 0.5f;
        assertEquals(expected, midpoint, 1e-4f);
        
        // End should be at end learning rate
        assertEquals(0.0001f, schedule.getLearningRate(10, 10), 1e-6f);
    }
    
    @Test
    public void testCombinedSchedule() {
        // Warmup then decay
        LearningRateSchedule warmup = LearningRateSchedule.linearWarmup(0.001f, 5);
        LearningRateSchedule decay = LearningRateSchedule.exponentialDecay(0.001f, 0.9f);
        LearningRateSchedule combined = LearningRateSchedule.combine(warmup, decay, 5);
        
        // During warmup
        assertTrue(combined.getLearningRate(0, 20) < 0.001f); // Starts low
        assertEquals(0.001f, combined.getLearningRate(4, 20), 1e-6f); // Reaches target
        
        // After switch - exponential decay starts
        float firstDecay = combined.getLearningRate(5, 20);
        assertEquals(0.001f * 0.9f * 0.9f * 0.9f * 0.9f * 0.9f, firstDecay, 1e-6f); // 0.9^5
        assertTrue(combined.getLearningRate(10, 20) < firstDecay); // Continues decaying
    }
}