package org.machine.learning.cart.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.machine.learning.cart.model.MinGINITuple;
import org.machine.learning.cart.utils.CARTUtils;
import org.machine.learning.cart.utils.DecisionTreeUtils;

public class CARTCore {

    /**
     * 计算最小的 Gini 值，及其值对应的属性名称
     * 
     * @param currentData
     * @return
     */
    public MinGINITuple<String, String, Double> minGiniMap(List<List<String>> currentData) {
        MinGINITuple<String, String, Double> result = new MinGINITuple<>();
        
        List<String> attributeNames = CARTUtils.getAttributeNames(currentData);
        
        double minGini = Double.MAX_VALUE;
        String minGiniAttributeName = "";
        String minGiniStatus = "";
        for (String attributeName : attributeNames) {
            Map<String, Double> gini = currentAttributeGini(currentData, attributeName);
            Set<String> keySet = gini.keySet();
            for (String key : keySet) {
                if (gini.get(key) < minGini) {
                    minGini = gini.get(key);
                    minGiniStatus = key;
                    minGiniAttributeName = attributeName;
                }
            }
        }
        
        result.setAttriuteName(minGiniAttributeName);
        result.setStatusName(minGiniStatus);
        result.setGiniValue(minGini);
        
        return result;
    }
    
    // TODO -------------------------------------------- private separated line ----------------------------------------------
    
    /**
     * 计算某一特征属性下，获得最小 GINI 指数时的状态信息
     * {恒温=0.5404761904761904}
     * 
     * @param currentData
     * @param attribute
     * @return
     */
    private Map<String, Double> currentAttributeGini(List<List<String>> currentData, String attribute) {
        Map<String, Map<String, Integer>> map = DecisionTreeUtils.getAttributeStatusMap(currentData, attribute);
        return currentAttributeMinGini(map);
    }
    
    /**
     * 计算某一特征属性下，获得最小 GINI 指数时的状态信息
     * {恒温=0.5404761904761904}
     * 
     * @param attributeMap
     * @return
     */
    private Map<String, Double> currentAttributeMinGini(Map<String, Map<String, Integer>> attributeMap) {
        Map<String, Double> result = new HashMap<>();
        Set<String> keySet = attributeMap.keySet();
        
        String minStatus = "";
        double minGini = Double.MAX_VALUE;
        
        for (String currentStatus : keySet) {
            double gini = currentAttributeDichotomyGINI(attributeMap, currentStatus, keySet);
            if (minGini >= gini) {
                minGini = gini;
                minStatus = currentStatus;
            }
        }
        
        result.put(minStatus, minGini);
        
        return result;
    }
    
    /**
     * 计算某一特征下，被某一状态二分化后的 GINI 指数
     * 
     * @param attributeMap
     * @param currentStatus
     * @param keySet
     * @return
     */
    private double currentAttributeDichotomyGINI(Map<String, Map<String, Integer>> attributeMap, String currentStatus, Set<String> keySet) {
        Map<String, Map<String, Integer>> attributeDichotomyMap = getAttributeDichotomyMap(attributeMap, currentStatus, keySet);
        int totalCount = 0;
        int positiveTotalCount = 0; // 正面的数据总数，比如：冷血={爬行类=3, 两栖类=2, 鱼类=3} 的总数为 8
        int negativeTotalCount = 0; // 反面的数据总数，比如：Negative={哺乳类=5, 鸟类=2} 的总数为 7
        
        // xxx
        Set<String> dichotomyKeySet = attributeDichotomyMap.keySet();
        for (String dichotomyKey : dichotomyKeySet) {
            Set<String> subDichotomyKeySet = attributeDichotomyMap.get(dichotomyKey).keySet();
            if (dichotomyKey.equals(currentStatus)) {
                for (String subDichotomyKey : subDichotomyKeySet) {
                    positiveTotalCount += attributeDichotomyMap.get(dichotomyKey).get(subDichotomyKey);
                }
            } else {
                for (String subDichotomyKey : subDichotomyKeySet) {
                    negativeTotalCount += attributeDichotomyMap.get(dichotomyKey).get(subDichotomyKey);
                }
            }
        }
        
        totalCount = positiveTotalCount + negativeTotalCount;
        
        // xxx
        double positiveGINI = 0.0;
        double negativeGINI = 0.0;
        for (String dichotomyKey : dichotomyKeySet) {
            Set<String> subDichotomyKeySet = attributeDichotomyMap.get(dichotomyKey).keySet();
            if (dichotomyKey.equals(currentStatus)) {
                for (String subDichotomyKey : subDichotomyKeySet) {
                    positiveGINI += (1.0 * attributeDichotomyMap.get(dichotomyKey).get(subDichotomyKey) / positiveTotalCount) * (1.0 * attributeDichotomyMap.get(dichotomyKey).get(subDichotomyKey) / positiveTotalCount);
                }
            } else {
                for (String subDichotomyKey : subDichotomyKeySet) {
                    negativeGINI += (1.0 * attributeDichotomyMap.get(dichotomyKey).get(subDichotomyKey) / negativeTotalCount) * (1.0 * attributeDichotomyMap.get(dichotomyKey).get(subDichotomyKey) / negativeTotalCount);
                }
            }
        }
        
        positiveGINI = 1 - positiveGINI;
        negativeGINI = 1 - negativeGINI;
        
        return (1.0 * positiveTotalCount / totalCount) * positiveGINI + (1.0 * negativeTotalCount / totalCount) * negativeGINI;
    }
    
    /**
     * 获得某一特征下，某一个状态的二分化
     * 
     * @param attributeMap
     * @param currentStatus
     * @param keySet
     * @return
     */
    private Map<String, Map<String, Integer>> getAttributeDichotomyMap(Map<String, Map<String, Integer>> attributeMap, String currentStatus, Set<String> keySet) {
        Map<String, Map<String, Integer>> attributeDichotomyMap = new HashMap<>();
        attributeDichotomyMap.put(currentStatus, attributeMap.get(currentStatus));
        String currentStatusNegative = "Negative";
        attributeDichotomyMap.put(currentStatusNegative, getAttributeDichotomyNegativeMap(attributeMap, currentStatus, currentStatusNegative, keySet));
        return attributeDichotomyMap;
    }
    
    /**
     * 获得某一特征属性下，某一个状态的反面状态信息
     * 比如：{哺乳类=5, 鸟类=2}
     * 
     * @param attributeMap
     * @param currentStatus
     * @param currentStatusNegative
     * @param keySet
     * @return
     */
    private Map<String, Integer> getAttributeDichotomyNegativeMap(Map<String, Map<String, Integer>> attributeMap, String currentStatus, String currentStatusNegative, Set<String> keySet) {
        Map<String, Integer> result = new HashMap<>();
        for (String negativeKey : keySet) {
            if (negativeKey.equals(currentStatus)) {
                continue;
            }
            
            Map<String, Integer> subAttributeMap = attributeMap.get(negativeKey);
            Set<String> classifyKeySet = subAttributeMap.keySet();
            for (String classifyKey : classifyKeySet) {
                if (!result.containsKey(classifyKey)) {
                    result.put(classifyKey, subAttributeMap.get(classifyKey));
                } else {
                    result.put(classifyKey, subAttributeMap.get(classifyKey) + result.get(classifyKey));
                }
            }
        }
        
        return result;
    }
}
