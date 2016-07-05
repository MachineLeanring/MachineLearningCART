package org.machine.learning.cart.model;

/**
 * <p>
 * 针对某一特征属性下，某一状态及其 GINI 值 比如：<体温, 恒温, 0.5404761904761904>
 * </p>
 * Create Date: 2016年7月4日 Last Modify: 2016年7月4日
 * 
 * @author <a href="http://weibo.com/u/5131020927">Q-WHai</a>
 * @see <a href="http://blog.csdn.net/lemon_tree12138">http://blog.csdn.net/
 *      lemon_tree12138</a>
 * @version 0.0.1
 */
public class MinGINITuple<T, M, N> {
    private T attriuteName = null;
    private M statusName = null;
    private N giniValue = null;

    public T getAttriuteName() {
        return attriuteName;
    }

    public void setAttriuteName(T attriuteName) {
        this.attriuteName = attriuteName;
    }

    public M getStatusName() {
        return statusName;
    }

    public void setStatusName(M statusName) {
        this.statusName = statusName;
    }

    public N getGiniValue() {
        return giniValue;
    }

    public void setGiniValue(N giniValue) {
        this.giniValue = giniValue;
    }

    @Override
    public String toString() {
        return String.join(", ", String.valueOf(attriuteName), String.valueOf(statusName), String.valueOf(giniValue));
    }
}
