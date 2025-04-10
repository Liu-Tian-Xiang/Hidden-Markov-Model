#pragma once

namespace HMM
{
    /**
     * A number that stores the natural log of its value (and a zero bit). Intented for use with extremely large or extremely small numbers, or in cases where fast division is
     * more important than high precision (division is implemented as subtraction of logs). Precision decreases with increasing value distance from e.
     * @warning Note that addition and subtraction can fail if the magnitudes of the numbers are significantly different (due to over/underflow of intermediate numbers)!
     * Use LnDouble::addBig if this becomes a problem (though it is slow and requires MPFR), or LnDouble::balancedAdd for lists of LnDouble (though this uses recursion
     * of depth lg(n) and can potentially blow the stack)
     */
    class LnDouble
    {
	public:
	    LnDouble();
	    LnDouble(double n);

	    LnDouble& operator=(const LnDouble& rhs);
	    LnDouble& operator+=(const LnDouble& rhs);
	    LnDouble& operator-=(const LnDouble& rhs);
	    LnDouble& operator*=(const LnDouble& rhs);
	    LnDouble& operator/=(const LnDouble& rhs);
	    const LnDouble operator+(const LnDouble& rhs) const;
	    const LnDouble operator-(const LnDouble& rhs) const;
	    const LnDouble operator*(const LnDouble& rhs) const;
	    const LnDouble operator/(const LnDouble& rhs) const;
	    LnDouble& operator=(const double& rhs);

	    //these comparisons deliberately do not check if |a-b|<epsilon - that's left up to the user
	    bool operator==(const LnDouble& rhs) const;
	    bool operator!=(const LnDouble& rhs) const;
	    bool operator==(const double& rhs) const;
	    bool operator!=(const double& rhs) const;
	    friend bool operator<(const LnDouble& lhs, const LnDouble& rhs);
	    friend bool operator>(const LnDouble& lhs, const LnDouble& rhs);
	    friend bool operator<=(const LnDouble& lhs, const LnDouble& rhs);
	    friend bool operator>=(const LnDouble& lhs, const LnDouble& rhs);

	    friend ostream& operator <<(ostream& os, const LnDouble& rhs);

	    static LnDouble pow(const LnDouble& base, const LnDouble& exponent);
	    /**
	     * Adds a list of LnDouble.
	     * @warning this uses recursion of depth lg(listSize), can potentially blow the stack with very long lists
	     * @warning the LnDouble in the list should be of similar magnitude, else the addition can potentially fail as intermediate numbers over/underflow
	     * @param list the list of LnDouble
	     * @param listSize the number of LnDouble in the list
	     * @return the sum
	     */
	    static LnDouble balancedAdd(const LnDouble* list, const int listSize);

#ifdef LNSNUM_ENABLE_ADD_BIG
	    /**
	     * Adds two LnDouble of significantly different magnitudes. Much slower than operator+ and requires MPFR, but avoids most if not all intermediate over/underflow issues.
	     * @param a one LnDouble
	     * @param b another LnDouble
	     * @return their sum
	     */
	    static LnDouble addBig(const LnDouble& a, const LnDouble& b);
#endif
	    /**
	     * Returns the natural log of this LnDouble's value (actually, the internal log representation of the value)
	     * @return log(this)
	     */
	    double getLog() const;
	    /**
	     * Returns the value of this LnDouble as a double.
	     * @warning this can easily over/underflow, check the log (see getLog()) against log(LDBL_MAX) and log(LDBL_MIN) to guard against this
	     */
	    double getVal() const;
	    /**
	     * Returns true if the value of this LnDouble is zero, otherwise returns false
	     * @return (*this == 0)
	     */
	    bool isZero() const;
	    /**
	     * Sets the LnDouble to e^n if it is nonzero.
	     * @warning if this is zero, setLog will have no effect! If you are unsure, you can use LnDouble l = 1 before l.setLog(n).
	     * @param n the natural log of the new value
	     */
	    void setLog(double n);
	private:
	    double num;
	    bool zero;
	    static LnDouble balancedAddHelper(const LnDouble* list, const int begin, const int end);

#ifdef LNSNUM_ENABLE_ADD_BIG
	    static mpfr_rnd_t rndMode;
	    static mpfr_prec_t defaultPrec;
#endif
    }; //class LnDouble

    inline LnDouble& LnDouble::operator+=(const LnDouble &rhs)
    {
	if (rhs.isZero()) {}
	else if (zero)
	{
	    num = rhs.getLog();
	    zero = false;
	}
	else
	{
	    num = rhs.getLog() + log(exp(num - rhs.getLog()) + 1);
	}
	return *this;
    }

    inline LnDouble& LnDouble::operator-=(const LnDouble &rhs)
    {
	if (rhs.isZero()) {}
	else if (zero) {/* implement if negative numbers are necessary */}
	else if (num == rhs.getLog())
	{
	    num = 0.0;
	    zero = true;
	}
	else num = rhs.getLog() + log(exp(num - rhs.getLog()) - 1);
	return *this;
    }

    inline LnDouble& LnDouble::operator*=(const LnDouble &rhs)
    {
	if (zero || rhs.isZero())
	{
	    zero = true;
	    num = 0.0;
	}
	else num += rhs.getLog();
	return *this;
    }

    inline LnDouble& LnDouble::operator/=(const LnDouble &rhs)
    {
	if (zero || rhs == 0.0)
	{
	    zero = true;
	    num = 0.0;
	}
	else num -= rhs.getLog();
	return *this;
    }

    inline const LnDouble LnDouble::operator+(const LnDouble &rhs) const
    {
	LnDouble result = *this;
	result += rhs;
	return result;
    }

    inline const LnDouble LnDouble::operator-(const LnDouble &rhs) const
    {
	LnDouble result = *this;
	result -= rhs;
	return result;
    }

    inline const LnDouble LnDouble::operator*(const LnDouble &rhs) const
    {
	LnDouble result = *this;
	result *= rhs;
	return result;
    }

    inline const LnDouble LnDouble::operator/(const LnDouble &rhs) const
    {
	LnDouble result = *this;
	result /= rhs;
	return result;
    }

    inline LnDouble& LnDouble::operator=(const LnDouble &rhs)
    {
	if (this != &rhs)
	{
	    if (rhs.isZero()) zero = true;
	    else zero = false;
	    num = rhs.getLog();
	}
	return *this;
    }

    inline LnDouble& LnDouble::operator=(const double &rhs)
    {
	if (rhs == 0.0)
	{
	    zero = true;
	    num = 0.0;
	}
	else
	{
	    zero = false;
	    num = log(rhs);
	}
	return *this;
    }
    inline double LnDouble::getLog() const {return num;}

    inline double LnDouble::getVal() const
    {
	if (zero) return 0.0;
	else return exp(num);
    }

    inline bool LnDouble::isZero() const {return zero;}

} // namespace HMM

namespace HMM
{
#ifdef LNSNUM_ENABLE_ADD_BIG
    mpfr_rnd_t LnDouble::rndMode = mpfr_get_default_rounding_mode();
    mpfr_prec_t LnDouble::defaultPrec = mpfr_get_default_prec();
#endif
    LnDouble::LnDouble()
	: num (0)
	  , zero (true)
    {}

    LnDouble::LnDouble(double n)
    {
	if (n == 0.0)
	{
	    num = 0;
	    zero = true;
	}
	else
	{
	    num = log(n);
	    zero = false;
	}
    }

    bool LnDouble::operator==(const LnDouble &rhs) const
    {
	if (zero != rhs.isZero()) return false;
	else if (zero && rhs.isZero()) return true;
	else if (num == rhs.getLog()) return true;
	else return false;
    }

    bool LnDouble::operator!=(const LnDouble &rhs) const
    {
	return !(*this == rhs);
    }

    bool LnDouble::operator==(const double &rhs) const
    {
	if (zero != (rhs == 0.0)) return false;
	else if (zero && rhs == 0.0) return true;
	else if (num == log(rhs)) return true;
	else return false;
    }

    bool LnDouble::operator!=(const double &rhs) const
    {
	return !(*this == rhs);
    }

    ostream& operator <<(ostream &os, const LnDouble &rhs)
    {
	os << rhs.getVal();
	return os;
    }

    LnDouble operator+(double lhs, LnDouble &rhs)
    {
	LnDouble lhsLnDouble(lhs);
	return lhsLnDouble + rhs;
    }

    LnDouble operator+(LnDouble &lhs, double &rhs)
    {
	return (rhs + lhs);
    }

    LnDouble operator-(double lhs, LnDouble &rhs)
    {
	LnDouble lhsLnDouble(lhs);
	return lhsLnDouble - rhs;
    }

    LnDouble operator-(LnDouble &lhs, double rhs)
    {
	LnDouble rhsLnDouble(rhs);
	return lhs - rhsLnDouble;
    }

    LnDouble operator*(double lhs, LnDouble &rhs)
    {
	LnDouble lhsLnDouble(lhs);
	return lhsLnDouble * rhs;
    }

    LnDouble operator*(LnDouble &lhs, double &rhs)
    {
	return (rhs * lhs);
    }

    LnDouble operator/(double lhs, LnDouble &rhs)
    {
	LnDouble lhsLnDouble(lhs);
	return lhsLnDouble / rhs;
    }

    LnDouble operator/(LnDouble &lhs, double rhs)
    {
	LnDouble rhsLnDouble(rhs);
	return lhs / rhsLnDouble;
    }

    bool operator<(const LnDouble &lhs, const LnDouble &rhs)
    {
	if (rhs.isZero()) return false;
	else if (lhs.isZero() && !rhs.isZero()) return true;
	else return (lhs.getLog() < rhs.getLog());
    }
    bool operator>(const LnDouble &lhs, const LnDouble &rhs)
    {
	if (lhs.isZero()) return false;
	else if (!lhs.isZero() && rhs.isZero()) return true;
	else return (lhs.getLog() > rhs.getLog());
    }
    bool operator<=(const LnDouble &lhs, const LnDouble &rhs)
    {
	if (lhs.isZero() && !rhs.isZero()) return true;
	else if (!lhs.isZero() && rhs.isZero()) return false;
	else if (lhs.isZero() && rhs.isZero()) return true;
	else return (lhs.getLog() <= rhs.getLog());
    }
    bool operator>=(const LnDouble &lhs, const LnDouble &rhs)
    {
	if (lhs.isZero() && !rhs.isZero()) return false;
	else if (!lhs.isZero() && rhs.isZero()) return true;
	else if (lhs.isZero() && rhs.isZero()) return true;
	else return (lhs.getLog() >= rhs.getLog());
    }
    bool operator<(const LnDouble &lhs, const double &rhs)
    {
	if (lhs.isZero()) return (0 < rhs);
	else return (lhs.getLog() < log(rhs));
    }
    bool operator>(const LnDouble &lhs, const double &rhs)
    {
	if (lhs.isZero()) return (0 > rhs);
	else return (lhs.getLog() > log(rhs));
    }
    bool operator<=(const LnDouble &lhs, const double &rhs)
    {
	if (lhs.isZero()) return (0 <= rhs);
	else return (lhs.getLog() <= log(rhs));
    }
    bool operator>=(const LnDouble &lhs, const double &rhs)
    {
	if (lhs.isZero()) return (0 >= rhs);
	else return (lhs.getLog() >= log(rhs));
    }
    bool operator<(const double &lhs, const LnDouble &rhs)
    {return (rhs > lhs);}
    bool operator>(const double &lhs, const LnDouble &rhs)
    {return (rhs < lhs);}
    bool operator<=(const double &lhs, const LnDouble &rhs)
    {return (rhs >= lhs);}
    bool operator>=(const double &lhs, const LnDouble &rhs)
    {return (rhs <= lhs);}

    LnDouble LnDouble::pow(const LnDouble &base, const LnDouble &exponent)
    {
	if (exponent.isZero())
	{
	    LnDouble ans(1);
	    return ans;
	}
	else if (base.isZero())
	{
	    LnDouble ans(0);
	    return ans;
	}
	else if (base.getLog() < 0.0)
	{
	    double temp = log(-base.getLog());
	    temp += exponent.getLog();
	    double temp2 = -exp(temp);
	    LnDouble ans(1);
	    ans.setLog(temp2);
	    return ans;
	}
	else
	{
	    double temp = log(base.getLog());
	    temp += exponent.getLog();
	    double temp2 = exp(temp);
	    LnDouble ans(1);
	    ans.setLog(temp2);
	    return ans;
	}
    }

#ifdef LNSNUM_ENABLE_ADD_BIG
    LnDouble LnDouble::addBig(const LnDouble& a, const LnDouble& b)
    {
	if (b.isZero()) {return a;}
	else if (a.isZero()) {return b;}
	else
	{
	    double result;
	    mpfr_t l;
	    mpfr_t r;
	    mpfr_t dif;
	    mpfr_t ans;
	    mpfr_t log;
	    mpfr_inits2(defaultPrec, l, r, dif, ans, log, (mpfr_ptr) NULL);
	    mpfr_set_d(l, a.getLog(), rndMode);
	    mpfr_set_d(r, b.getLog(), rndMode);

	    mpfr_sub(dif, l, r, rndMode);
	    mpfr_exp(ans, dif, rndMode);
	    mpfr_add_ui(dif, ans, (unsigned long int) 1, rndMode);
	    mpfr_log(log, dif, rndMode);
	    mpfr_add(ans, log, r, rndMode);
	    result = mpfr_get_d(ans, rndMode);
	    mpfr_clears(l, r, dif, ans, log, (mpfr_ptr) NULL);
	    LnDouble res(1);
	    res.setLog(result);
	    return res;
	}
    }
#endif

    void LnDouble::setLog(double n)
    {
	num = n;
    }

    LnDouble LnDouble::balancedAdd(const LnDouble* list, const int listSize)
    {
	if (listSize == 0) return 0;
	else return LnDouble::balancedAddHelper(list, 0, listSize - 1);
    }

    LnDouble LnDouble::balancedAddHelper(const LnDouble* list, const int begin, const int end)
    {
	if (begin == end) return list[begin];
	else
	{
	    int midpoint = (end + begin) / 2;
	    return LnDouble::balancedAddHelper(list, begin, midpoint) + LnDouble::balancedAddHelper(list, midpoint + 1, end);
	}
    }

}

