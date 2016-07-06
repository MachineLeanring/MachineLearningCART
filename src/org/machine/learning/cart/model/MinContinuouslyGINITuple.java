package org.machine.learning.cart.model;

/**
 * <p>
 * 最小连续变量的 Tuple
 * </p>
 * Create Date: 2016年7月5日
 * Last Modify: 2016年7月5日
 * 
 * @author <a href="http://weibo.com/u/5131020927">Q-WHai</a>
 * @see <a href="http://blog.csdn.net/lemon_tree12138">http://blog.csdn.net/lemon_tree12138</a>
 * @version 0.0.1
 */
public class MinContinuouslyGINITuple<T, K> {

    private double minGINI = Double.MAX_VALUE;
    private int minGINIIndex = 0;
    
    public MinContinuouslyGINITuple() {
    }
    
    public MinContinuouslyGINITuple(double minGINI, int minGINIIndex) {
        this.minGINI = minGINI;
        this.minGINIIndex = minGINIIndex;
    }

    public double getMinGINI() {
        return minGINI;
    }

    public void setMinGINI(double minGINI) {
        this.minGINI = minGINI;
    }

    public int getMinGINIIndex() {
        return minGINIIndex;
    }

    public void setMinGINIIndex(int minGINIIndex) {
        this.minGINIIndex = minGINIIndex;
    }
    
}
