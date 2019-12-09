def shapiro_test(data, alpha):
    from scipy.stats import shapiro
    
    stat, p = shapiro(data)
    print("Statistics: %.3f \np-value: %.3f" % (stat, p))
    if p > alpha:
        print("Sample looks Normal. [Fail to reject H0.]")
    else:
        print("Sample does not look Normal. [Reject H0.]")

    return p

def agostino_test(data, alpha):
    from scipy.stats import normaltest
    
    stat, p = normaltest(data)
    print("Statistics: %.3f \np-value: %.3f" % (stat, p))
    if p > alpha:
        print("Sample looks Normal. [Fail to reject H0.]")
    else:
        print("Sample does not look Normal. [Reject H0.]")
        
    return p