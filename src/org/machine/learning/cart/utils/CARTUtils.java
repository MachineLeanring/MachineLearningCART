package org.machine.learning.cart.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.core.utils.str.StringUtils;
import org.machine.learning.cart.model.ContinuouslyVariableUnit;
import org.machine.learning.cart.model.MinContinuouslyGINITuple;
import org.machine.learning.cart.model.MinGINITuple;

public class CARTUtils {

    /**
     * 转变原始数据中的连续变量为离散变量
     * 
     * @param data
     */
    public static void transformContinuouslyVariables(List<List<String>> data) {
        List<Integer> indexs = getContinuouslyVariableIndexs(data);
        for (Integer index : indexs) {
            List<ContinuouslyVariableUnit> units = extractionContinuouslyVariableUnits(data, index);
            units.sort(new ContinuouslyVariableUnitComparatorByValue());
            int threshold = units.get(getAttributeThreshold(units).getMinGINIIndex()).getValue();
            transformContinuouslyVariablesByThreshold(data, index, threshold);
        }
    }
    
    /**
     * 根据最小 GINI 指数进行原数据转换
     * 
     * @param data
     * @param minGINITuple
     */
    public static void transformAttruteStatus(List<List<String>> data, MinGINITuple<String, String, Double> minGINITuple) {
        List<String> attributeNames = getAttributeNames(data);
        int findIndex = 0;
        for (int i = 0; i < attributeNames.size(); i++) {
            if (attributeNames.get(i).equals(minGINITuple.getAttriuteName())) {
                findIndex = i + 1;
            }
        }
        
        for (int i = 1; i < data.size(); i++) {
            List<String> rowData = data.get(i);
            if (!rowData.get(findIndex).equals(minGINITuple.getStatusName())) {
                rowData.set(findIndex, "Negative");
            }
        }
    }
    
    /**
     * 获得所有的特征属性列表
     * 
     * @param currentData
     * @return
     */
    public final static List<String> getAttributeNames(List<List<String>> currentData) {
        List<String> attributeNames = new ArrayList<>();
        List<String> getAttributeNames = currentData.get(0);
        for (int attributeIndex = 1; attributeIndex < getAttributeNames.size() - 1; attributeIndex++) {
            attributeNames.add(getAttributeNames.get(attributeIndex));
        }
        
        return attributeNames;
    }
    
    /**
     * 获得当前数据下的所有结果分类列表
     * 
     * @param currentData
     * @return
     */
    public final static Set<String> getClassifyList(List<List<String>> currentData) {
        Set<String> classifyList = new HashSet<>();
        for (int rowIndex = 1; rowIndex < currentData.size(); rowIndex++) {
            List<String> rowData = currentData.get(rowIndex);
            classifyList.add(rowData.get(rowData.size() - 1));
        }
        
        return classifyList;
    }
    
    /**
     * 计算连续变量的阈值
     * 
     * @param units
     * @return
     */
    public static MinContinuouslyGINITuple<Integer, Double> getAttributeThreshold(List<ContinuouslyVariableUnit> units) {
        MinContinuouslyGINITuple<Integer, Double> minGINITuple = new MinContinuouslyGINITuple<>();
        double minGINI = Double.MAX_VALUE;
        int minGINIIndex = 0;
        for (int index = 0; index < units.size() - 1; index++) {
            double gini = getGINIByThreshold(units, index);
            if (minGINI > gini) {
                minGINI = gini;
                minGINIIndex = index;
            }
        }
        
        minGINITuple.setMinGINI(minGINI);
        minGINITuple.setMinGINIIndex(minGINIIndex);
        
        return minGINITuple;
    }
    
    // TODO -------------------------------------------- private separated line ----------------------------------------------
    
    /**
     * 获得连续变量的所有下标
     * 
     * @param data
     * @return
     */
    private static List<Integer> getContinuouslyVariableIndexs(List<List<String>> data) {
        List<Integer> indexs = new ArrayList<>();
        
        List<String> lastRowData = data.get(data.size() - 1);
        for (int index = 1; index < lastRowData.size() - 1; index++) {
            if (StringUtils.RegexUtils.isNumberString(lastRowData.get(index))) {
                indexs.add(index);
            }
        }
        
        return indexs;
    }
    
    /**
     * 根据阈值修改原始数据中第 index 列，将连续变量修改为离散变量
     * 
     * @param data
     * @param index
     * @param threshold
     */
    private static void transformContinuouslyVariablesByThreshold(List<List<String>> data, int index, int threshold) {
        for (int i = 1; i < data.size(); i++) {
            List<String> rowData = data.get(i);
            if (Integer.parseInt(rowData.get(index)) <= threshold) {
                rowData.set(index, "<=" + threshold);
            } else {
                rowData.set(index, ">" + threshold);
            }
        }
    }
    
    /**
     * 抽取原数据中某一列的全部连续变量单元
     * 
     * @param data
     * @param index
     * @return
     */
    private static List<ContinuouslyVariableUnit> extractionContinuouslyVariableUnits(List<List<String>> data, int index) {
        List<ContinuouslyVariableUnit> units = new ArrayList<>();
        for (int rawIndex = 1; rawIndex < data.size(); rawIndex++) {
            ContinuouslyVariableUnit unit = new ContinuouslyVariableUnit();
            List<String> singleRowData = data.get(rawIndex);
            unit.setValue(Integer.parseInt(singleRowData.get(index)));
            unit.setClassify(singleRowData.get(singleRowData.size() - 1));
            units.add(unit);
        }
        
        return units;
    }
    
    /**
     * 通过某一个阈值计算 Info
     * 
     * @param units
     * @param splitIndex
     * @return
     */
    private static double getGINIByThreshold(List<ContinuouslyVariableUnit> units, int splitIndex) {
        Set<String> classifySet = getContinuouslyVariableUnitClassifySet(units);
        Map<String, Integer> leftMap = getContinuouslyVariableUnitMap(units, 0, splitIndex, classifySet);
        Map<String, Integer> rightMap = getContinuouslyVariableUnitMap(units, splitIndex + 1, units.size() - 1, classifySet);
        
        return infoBySplitThreshold(leftMap, rightMap);
    }
    
    /**
     * 获得所有结果集
     * 
     * @param units
     * @return
     */
    private static Set<String> getContinuouslyVariableUnitClassifySet(List<ContinuouslyVariableUnit> units) {
        Set<String> set = new HashSet<>();
        for (ContinuouslyVariableUnit unit : units) {
            set.add(unit.getClassify());
        }
        return set;
    }
    
    /**
     * 统计被阈值分开后的结果分布
     * 
     * @param units
     * @param startIndex
     * @param endIndex
     * @param classifySet
     * @return
     */
    private static Map<String, Integer> getContinuouslyVariableUnitMap(List<ContinuouslyVariableUnit> units, int startIndex, int endIndex, Set<String> classifySet) {
        // TODO
        Map<String, Integer> unitMap = new HashMap<>();
        for (String classify : classifySet) {
            unitMap.put(classify, 0);
        }
        
        for (int index = startIndex; index <= endIndex; index++) {
            if (unitMap.containsKey(units.get(index).getClassify())) {
                unitMap.put(units.get(index).getClassify(), unitMap.get(units.get(index).getClassify()) + 1);
            } else {
                unitMap.put(units.get(index).getClassify(), 1);
            }
        }
        
        return unitMap;
    }
    
    /**
     * 计算被某一个阈值分隔后的 Info
     * 
     * @param leftMap
     * @param rightMap
     * @return
     */
    private static double infoBySplitThreshold(Map<String, Integer> leftMap, Map<String, Integer> rightMap) {
        int[] leftValues = getContinuouslyVariableUnitCounts(leftMap);
        int[] rightValues = getContinuouslyVariableUnitCounts(rightMap);
        return gini(leftValues, rightValues);
    }
    
    /**
     * 获得被某一个阈值分隔后的结果分布数组
     * [5, 9]
     * 
     * @param unitMap
     * @return
     */
    private static int[] getContinuouslyVariableUnitCounts(Map<String, Integer> unitMap) {
        int[] unitCounts = new int[unitMap.size() == 1 ? 2 : unitMap.size()];
        Iterator<Map.Entry<String, Integer>> iterator = unitMap.entrySet().iterator();
        for (int i = 0; i < unitCounts.length; i++) {
            if (iterator.hasNext()) {
                unitCounts[i] = iterator.next().getValue();
            }
        }
        
        if (unitMap.size() == 1) {
            unitCounts[1] = 0;
        }
        
        return unitCounts;
    }
    
    private static double gini(int[] left, int[] right) {
        double totalCount = 0.0;
        double leftCount = 0.0;
        double rightCount = 0.0;
        for (int value : left) {
            leftCount += value;
        }
        for (int value : right) {
            rightCount += value;
        }
        totalCount = leftCount + rightCount;
        
        return (leftCount / totalCount) * gini(left[0], left[1]) + (rightCount / totalCount) * gini(right[0], right[1]);
    }
    
    private static double gini(int a, int b) {
        int totalCount = a + b;
        return 1 - (Math.pow((1.0 * a / totalCount), 2) + Math.pow((1.0 * b / totalCount), 2));
    }
}
