# advanced C++

为了这门课的学习需要一些C++进阶的知识，包括并发和智能指针的部分，这些之前用的很少，因此需要学习一下。

### 右值引用

> 参考资料：[对C++11中的`移动语义`与`右值引用`的介绍与讨论 - 张浮生 - 博客园 (cnblogs.com)](https://www.cnblogs.com/neooelric/p/10878394.html)

```cpp
void func(int& x){cout<<"hello"<<endl;}
int main()
{
    int x=0;
    int y=0;
    func(x);	正确
    func(x+y);	错误
    func(1);	错误
}
```

由上述代码我们可以看到，我们的引用只能对左值进行引用，所谓左值我们可以用这个值能不能出现在等式的左边来判断。

如果将func函数的参数改为 `int&& x`​ 这后面两个语句就可以执行，第一条反而不能执行。一般来说，当函数参数中出现引用的时候，多数时候也需要一个右值引用的函数作为重载辅助。

而右值引用有一个特点，也即其对象往往是一个临时的变量，因此当我们在做拷贝构造函数的时候，我们也同样需要一个右值的重载，但是对于这样一个函数而言，像一般的深拷贝那样，new一块内存拷贝进去是一种相当的浪费，我们往往会这样做（Most typically, this rvalue will be a literal or temporary value. 字面值或者临时变量）

```cpp
string(string&& that)   // string&& is an rvalue reference to a string
    {
        data = that.data;
        that.data = nullptr;
    }
```

假设string内部是用一个data指针管理数组，我们完全不需要new一块内存再一一做拷贝，我们只需要把这块内存拿过来，并且将临时变量的那部分修改成nullptr，因为其之后马上会被销毁。

右值引用的本质是创建一个引用，来延长临时对象的生命周期 ？

```cpp
int&& x {2};
x=10;
cout<<x<<endl; 输出10
```

上述这个例子看似有点奇怪，似乎我们是对这个字面值做了引用，其实应该理解为对`int(2)`​这个临时变量做了右值引用

对于这些右值，其实我们可以用const的左值引用来接收

有一个比较tricky的现象是：**右值引用本身是左值**

### **移动语义和智能指针**

> 参考资料：[22.1 — Introduction to smart pointers and move semantics – Learn C++ (learncpp.com)](https://www.learncpp.com/cpp-tutorial/introduction-to-smart-pointers-move-semantics/)

需要智能指针的原因

* 当变量离开作用域的时候会调用析构函数，但在指针离开的时候却不会自动调用delete
* 深度拷贝的过程是浪费的

移动语义(move semantics)的本质在于转移所有权(ownership)，而不是做拷贝或赋值，C++的要求是尽可能地高性能，对于左值才做拷贝，对于右值仅做移动

```cpp
Auto_ptr2(Auto_ptr2& a) // note: not const
	{
		m_ptr = a.m_ptr; // transfer our dumb pointer from the source to our local object
		a.m_ptr = nullptr; // make sure the source no longer owns the pointer
	}

	// An assignment operator that implements move semantics
	Auto_ptr2& operator=(Auto_ptr2& a) // note: not const
	{
		if (&a == this)
			return *this;

		delete m_ptr; // make sure we deallocate any pointer the destination is already holding first
		m_ptr = a.m_ptr; // then transfer our dumb pointer from the source to the local object
		a.m_ptr = nullptr; // make sure the source no longer owns the pointer
		return *this;
	}
```

上述代码是一个我们简写的智能指针的拷贝和赋值重写，其核心思想是，智能指针的赋值和拷贝，并不会造成其所指向对象的拷贝，所指向的对象永远只有一份，这份代码的实现类似与早期C++中的 `std::auto_ptr`​，它有很多的问题，例如在上述实现中，你会发现如果你把智能指针传入函数中（发生拷贝构造），原本的智能指针居然在发生拷贝后无效了，这显然是不能接受的。

上述指针的问题在于，只能发生移动，不能发生拷贝，而有些智能指针需要兼顾二种功能。而当C++11引入右值引用之后，就可以做到。

对于函数返回值而言

* The C++ specification has a special rule that says automatic objects returned from a function by value can be moved even if they are l-values.

移动语义的应用场合

* 函数返回一个大的对象时，这个对象生命周期在函数内部
* 交换两个对象的内容，实际上只需要交换所有权就可以，但是因为这种场合下两个对象往往不是右值，因此我们需要使用`std::move`​​，其可以将**左值静态转换成右值**，即可触发移动构造函数和移动赋值函数  

  ```cpp
  #include <utility> // for std::move
  template<class T>
  void myswapMove(T& a, T& b)
  {
  	T tmp { std::move(a) }; // invokes move constructor
  	a = std::move(b); // invokes move assignment
  	b = std::move(tmp); // invokes move assignment
  }
  ```
* 任何可以做移动代替拷贝的情况下，都应该采用移动提高效率，例如`vec.push_back(std::move(x));`​做移动而不做拷贝，效率更高。但是要注意的是被move的这个变量不能再调用任何有关于其内容的方法，直到其被再次赋值。

### 标准库的智能指针

智能指针存在的首要目标是保证维护的对象在离开智能指针管理的范围之后正确delete。因此，我们不应该对智能指针动态分配内存

```cpp
#include <memory> // for std::unique_ptr
#include <utility> // for std::move
std::unique_ptr<Resource> res1{ new Resource() };
std::unique_ptr<Resource> res2{}; // Start as nullptr
res2 = std::move(res1); // res2 assumes ownership, res1 is set to null
```

这些智能指针都重载指针相应的操作符 Operator* returns a reference to the managed resource, and operator-> returns a pointer.

当未给智能指针分配对象或者是对象被mover给了其他的指针的时候，其设计保证此时的智能指针表现为空指针的性质

```cpp
if (res) // use implicit cast to bool to ensure res contains a Resource
	std::cout << *res << '\n'; // print the Resource that res is owning
```

现代智能指针都可以管理单个对象和数组对象，但是对于数组而言std::array and std::vector几乎永远是更好的选择

**智能指针的创建**

在C++14之后，我们会更倾向于用一些特定的函数来创造智能指针而不是像上面的例子一样直接new，tips：直接new的写法会存在一些较为特殊的情况发生内存泄露

```cpp
// Create a single dynamically allocated Fraction with numerator 3 and denominator 5
// We can also use automatic type deduction to good effect here
auto f1{ std::make_unique<Fraction>(3, 5) };
// Create a dynamically allocated array of Fractions of length 4
auto f2{ std::make_unique<Fraction[]>(4) };

auto f3{std::make_shared<Fraction>(3,5)};
```

**智能指针作为函数返回值**

在正常的设计下，一般该种情况比较少见

```cpp
std::unique_ptr<Resource> createResource()
{
    return std::make_unique<Resource>();
}
auto ptr{ createResource() };
```

**智能指针作为函数的参数**

要注意的是，std::unique_ptr不支持拷贝操作，其传值的时候只能依赖std::move或者智能指针本身的引用。或者还有一种方式是调用智能指针的get()方法，将资源本身的指针传入函数。

对于std::unique_ptr 不要把一份资源传给两个指针，编译器不会阻止（？？？）但是会造成泄露，如果使用make_unique就不会有上述问题

```cpp
Resource* res{ new Resource() };
std::unique_ptr<Resource> res1{ res };
std::unique_ptr<Resource> res2{ res };
```

**std::shared_ptr**

与unique的指针不同shared_ptr支持多个智能指针指向同一资源目标的情况，也即支持拷贝语义。但是需要明确的是，拷贝语义需要在发生在智能指针之间，而不是资源之间

```cpp
错误做法
std::shared_ptr<Resource> ptr1 { res };
std::shared_ptr<Resource> ptr2 { res };
正确做法
std::shared_ptr<Resource> ptr1{ res };
std::shared_ptr<Resource> ptr2 { ptr1 };
```

unique_ptr可以较为安全地转换成一个shared_ptr，反之则不然
