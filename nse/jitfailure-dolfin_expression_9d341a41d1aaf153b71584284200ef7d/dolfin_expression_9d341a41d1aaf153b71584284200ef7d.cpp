
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_9d341a41d1aaf153b71584284200ef7d : public Expression
  {
     public:
       

       dolfin_expression_9d341a41d1aaf153b71584284200ef7d()
       {
            _value_shape.push_back(2);
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] =  tensor([ 0.2070, -1.0588, -1.4552, -3.6224,  8.5159, -4.3239,  5.6020, -6.4518,
         9.7344, -8.3892, -8.4674, -0.5830, -4.8345,  2.7297,  6.4460, -5.4421,
         0.0200,  7.8300,  8.1464,  8.1246], grad_fn=<SelectBackward0>) * 0.05 * ( x[1] - 0.2 )/ 0.05  ;
          values[1] =  -1* tensor([ 0.2070, -1.0588, -1.4552, -3.6224,  8.5159, -4.3239,  5.6020, -6.4518,
         9.7344, -8.3892, -8.4674, -0.5830, -4.8345,  2.7297,  6.4460, -5.4421,
         0.0200,  7.8300,  8.1464,  8.1246], grad_fn=<SelectBackward0>) * 0.05 * ( x[0] - 0.2 )/ 0.05  ;

       }

       void set_property(std::string name, double _value) override
       {

       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {

       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_9d341a41d1aaf153b71584284200ef7d()
{
  return new dolfin::dolfin_expression_9d341a41d1aaf153b71584284200ef7d;
}

