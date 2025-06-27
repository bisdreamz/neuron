package dev.neuronic.net.serialization;

import dev.neuronic.net.activators.Activator;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.Optimizer;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

/**
 * Registry for custom serializable types.
 * 
 * Allows registration of custom activators, layers, and optimizers
 * without requiring reflection or hardcoded type IDs.
 */
public final class SerializationRegistry {
    
    // Factory interfaces for different component types
    public interface ActivatorFactory {
        Activator create();
    }
    
    public interface LayerFactory {
        Layer create(DataInputStream in, int version) throws IOException;
    }
    
    public interface OptimizerFactory {
        Optimizer create(DataInputStream in, int version) throws IOException;
    }
    
    // Registries for custom types
    private static final Map<String, ActivatorFactory> activatorFactories = new ConcurrentHashMap<>();
    private static final Map<String, LayerFactory> layerFactories = new ConcurrentHashMap<>();
    private static final Map<String, OptimizerFactory> optimizerFactories = new ConcurrentHashMap<>();
    
    // Reverse lookup for serialization
    private static final Map<Class<?>, String> classToName = new ConcurrentHashMap<>();
    
    private SerializationRegistry() {} // Prevent instantiation
    
    /**
     * Register a custom activator type.
     * Activators should be stateless singletons.
     */
    public static void registerActivator(String className, ActivatorFactory factory) {
        activatorFactories.put(className, factory);
        classToName.put(factory.create().getClass(), className);
    }
    
    /**
     * Register a custom layer type.
     */
    public static void registerLayer(String className, LayerFactory factory) {
        layerFactories.put(className, factory);
    }
    
    /**
     * Register a custom optimizer type.
     */
    public static void registerOptimizer(String className, OptimizerFactory factory) {
        optimizerFactories.put(className, factory);
    }
    
    /**
     * Get registered class name for a component, or null if not registered.
     */
    public static String getRegisteredName(Object component) {
        return classToName.get(component.getClass());
    }
    
    /**
     * Check if an activator class is registered.
     */
    public static boolean isActivatorRegistered(String className) {
        return activatorFactories.containsKey(className);
    }
    
    /**
     * Check if a layer class is registered.
     */
    public static boolean isLayerRegistered(String className) {
        return layerFactories.containsKey(className);
    }
    
    /**
     * Check if an optimizer class is registered.
     */
    public static boolean isOptimizerRegistered(String className) {
        return optimizerFactories.containsKey(className);
    }
    
    /**
     * Create a registered activator instance.
     */
    public static Activator createActivator(String className) {
        ActivatorFactory factory = activatorFactories.get(className);
        if (factory == null) {
            throw new IllegalArgumentException("Unknown registered activator: " + className);
        }
        return factory.create();
    }
    
    /**
     * Create a registered layer instance.
     */
    public static Layer createLayer(String className, DataInputStream in, int version) throws IOException {
        LayerFactory factory = layerFactories.get(className);
        if (factory == null) {
            throw new IllegalArgumentException("Unknown registered layer: " + className);
        }
        return factory.create(in, version);
    }
    
    /**
     * Create a registered optimizer instance.
     */
    public static Optimizer createOptimizer(String className, DataInputStream in, int version) throws IOException {
        OptimizerFactory factory = optimizerFactories.get(className);
        if (factory == null) {
            throw new IllegalArgumentException("Unknown registered optimizer: " + className);
        }
        return factory.create(in, version);
    }
}