---
title: Automatic Differentiation
date:   2020-06-11 16:39:48 +0530
categories: [deeplearning, mathematics]
tags: deeplearning autograd
math: true
toc: false
---

If you have ever used [pytorch](https://www.pytorch.org/) or [tensorflow](https://www.tensorflow.org/), you must have realised you never write the gradients! Isn't gradient backpropagation the soul?
This is where the smart people in deep learning frameworks do the sorcery. Automatic differentiation allows rapid development of complex architectures or, for peoople like me, tweaking around pre-built networks. Almost every other deep learning framework today uses it. You write the forward pass and *poof*, that's it, the magic flows. I have been trying to [implement](https://www.github.com/khizirsiddiqui/auto-grad) one for a few days now, and will try to put it into words here.

Automatic Differentiation comes in two flavors - forward mode and the reverse mode. Forward mode extensively uses *dual numbers* as the prime potion ingridient. Reverese mode is more widely used (even by [pytorch](https://www.pytorch.org/) or [tensorflow](https://www.tensorflow.org/)).

## Dual Numbers

Dual Numbers are hypercomplex numbers of form \\(a + b\epsilon\\) and extend the real numbers, pretty much like complex numbers, where \\(a\\) and \\(b\\) are real numbers and \\(\epsilon^2 = 0\\). This isn't new, [William Clifford](https://en.wikipedia.org/wiki/William_Kingdon_Clifford) introduced them in 1873. Remember *calculus* class? [Taylor Series](https://en.wikipedia.org/wiki/Taylor_series)(?) of a function \\(f(x)\\) around the point \\(x=a\\) :

$$ f(x) = \sum_{k=0}^{\infty}\frac{f^{(k)}(a)}{k!}(x-a)^k = f(a) + f'(a)(x-a) + \sum_{k=2}^{\infty}\frac{f^{(k)}(a)}{k!}(x-a)^k $$

So we will find some x where the function retains the value of \\(f'(x)\\) and discard all other higher degree derivatives. Clearly the real world (numbers) is of no help and here, we trick the equation using dual numbers. We will plug in \\(x = a + b\epsilon\\) in \\(f(x)\\) with \\(\epsilon \ne 0\\) while \\(\epsilon^2 = 0\\). Yes, we are collecting our potion ingredients.

$$ f(a + b\epsilon) = f(a) + f'(a)b\epsilon + \sum_{k=2}^{\infty}\frac{f^{(k)}(a)}{k!}b^k\epsilon^k $$

$$ = f(a) + f'(a)b\epsilon + \underbrace{\epsilon^2\sum_{k=2}^{\infty}\frac{f^{(k)}(a)}{k!}b^k\epsilon^{k-2}}_{0} $$

$$ = f(a) + f'(a)b\epsilon $$

We will set \\(b = 1\\), and we get

$$ f(a + \epsilon) = f(a) + f'(a)\epsilon $$

We get the dual number with real part \\(f(a)\\) as function value and the dual part \\(f'(a)\\) (like we call imaginary part in complex numbers). We have the function value as real part and the derivative at that point as dual part. But how is this usefull? Let's define and perform some operations on it.

### Operation on Dual Numbers

We define addition and multiplication operations in this space:

*Addition* : \\((a + b\epsilon) + (c + d\epsilon) = (a + c) + (b + d)\epsilon\\)

*Multiplication* : \\((a + b\epsilon) (c + d\epsilon) = (ac) + (bc + ad)\epsilon\\)

The dual number does not interfere with the real part, and *automatically* provides us with the derivative. This gets more clear when a and c are univariate (and even multivariate) functions -

$$ f(x + \epsilon) + g(x + \epsilon) = (f(x) + g(x)) + (f'(x) + g'(x))\epsilon $$

$$ f(x + \epsilon) \cdot g(x + \epsilon) = (f(x) \cdot g(x)) + (f'(x) \cdot g(x) + f(x) \cdot g'(x))\epsilon $$

Now we can derive more complex operations on the dual numbers with the same definition: when \\(f(x) = sin(x)\\) then

$$ f(x + \epsilon) = sin(x + \epsilon) = sin(x) + \frac{d}{dx}sin(x) \epsilon $$

$$ = sin(x) + cos(x) \epsilon $$

When \\(g(x) = e^x \\)

$$ g(x + \epsilon) = e^{x + \epsilon} = e^x + e^x \epsilon $$

When \\(h(x, y) = x / y \\)

$$ h(x + a\epsilon, y + b\epsilon) = x/y - \frac{xb - ya}{y^2}\epsilon $$

## Forward Mode Automatic Differentiation

Dual Numbers is the most common way to brew the forward flavor. It applies Leibniz's chain rule to each operation executed, multiplying the previous derivative with current and so on. This turns out to be fairly simple for hand calculations and even programming. Evaluate the expression and store the derivatives at the same time, fairly easy, right?

```python
class DualNumber:
    def ___init___(self, real, dual):
        self.real = real
        self.dual = dual

    def ___add___(self, other):
        return DualNumber(self.real + other.real, self.dual + other.dual)
```

This shall extend to ``___sub__``_,`` ___div___``,`` ___mul___`` and others for the full list see this [part of code](https://github.com/khizirsiddiqui/auto-grad/blob/master/dualnumber/DualNumber.py).

Defining the complex operations is also easy:
```python
def sin(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.sin(x.real), math.cos(x.real) * x.dual)
    else:
        return math.sin(x)
```
You can define a few [like this](https://github.com/khizirsiddiqui/auto-grad/blob/master/dualnumber/DualMath.py), and it should work wonders already.

### Complexity

Given a function \\(f : \mathbb{R}^n \rightarrow \mathbb{R}^m \\), two cases arise:

 - \\(m >> n\\) : Input dimension is smaller than the output dimension, the operations expand the dimension and should be computationally too much expensive.
 - \\(m << n\\) : A particular problem arises when the output dimension is less than the input dimensions, the complexity tends to increase extraordinarily, and we come up with better ways: reverse automatic differentiation.

## Reverse Mode Automatic Differentiation

This is what our well-known **backpropagation**.
