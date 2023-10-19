# oop

## object oriented programming in java

### interface inheritance

There is **hierarchy**(等级差异) between class and class.

For example, SLList and AList are two type of List with different data organizing way.  
They are **hyponyms** of List and List is **hypernyms** of them. We also called this relationship as **subclass and superclass**.

In java, the superclass is used to provide interface, and the subclass to provide implementation.  
Here is an example

```java
public interface List<Item>{...}
public AList<Item> implements List<Item> {...}
//this implements is a promise that we will cover all
//attributes and behaviors in the List<Item>
```

Inheritance and the function overriding, we solve many problems of function overload. We don't need to give each subclass a overloaded function.

**Note: ​**we don't need to write the "public" or "private" in interface.

**overriding**

When we are implementing these functions, we'd better note here @override

```java
@Override
public void addFirst(Item x){
	insert(x,0);
}
```

Inheritance can be multi-generational. One superclass may also is subclass of another class.

**default**

Sometime, we don't want the superclass only contians the interface, we hope it can complete some implementations.

```java
public interface List<Item>{
	...
	default public void print(){
			for(int i=0;i<size();i++){
				System.out.print(get(i)+" ");
			}
			System.out.println();
	}
}
```

For every subclass of List<Item>, if they don't make differenet implementations of this this interface.  
We will use this function.

For SLList, this default function is obviously an awful solution, so it can make its own implementation, and system  
will use its method instead of default one.

‍

### upcast and compile time type check

We have talked about the **rule of equals(a=b)** in java, which is the copy the bits from b to a.

Now, we refer to the superclass and subclass. Like below

```java
public interface List<Item>{...}
public AList<Item> implements List<Item> {...}
...
public static void func(List<int> list);
...
Alist<int> list;
func(list);  //what happened in the argument passing
```

Or we just use a more simple example

```java
List<Integer> somelist=new AList<Integer>();
```

While compiling, system will check  "is-a" relationship. In the example above, the AList is a List, so this sentence is right.  
However, if we swap their positions, List isn't an AList, and it will occur mistake.

### dynamic method selection

We still talk about the example below

```java
List<int> list=new AList<>();
```

List<int> is the **static type** list, we know all of reference type variables in java is pointer. As a pointer, list's type is List<int>

Besides, we actually create a AList object, and list points at an AList object. So AList<int> is list's **dynamic type.**

The dynamic type of the object can be changed, it can turned from AList to SLList. When program runs the **overidden** method, it will find the appropriate method in the dynamic type.

‍

### overload of inheritance

We have known that Java select the overloaded functions by type of parameter we give.

In fact, it will check the **static type ​**instead of dynamic type. Here we give an example

```java
//overloaded function
public static void func(List<Integer> list);
public static void func(SLList<Integer> list);
...
//in the main function
SLList<Integer> a=new SLList<>(); //static type is SLList and dynamic type is SLList too
List<Interger> b=a;				  //static type is List and dynamic type is SLList
func(a);						  //use the second one
func(b);						  //use the first one
```

### extends and implement

We have discussed the implementation relationship, which asks the subclass should implement all of methods the superclass  
provides.

However, sometimes we want to add more methods in the subclass, and in this situation we will use the keyword **extends .**

```java
public class RotatingSLList<Item> extends SLList<Item> {...}
```

In this example above, RotatingSLList will inherit all methods the SLList has and can make modifications or add some methods.

And we need to know what exactly do we inherit by extending.

1. All instances and static variables
2. All methods
3. All nested class

But we should notice that the **constructor** isn't inherited and we can't access the private members **directly ​**in subclass.

**The situation of constructor** is a little special. For ensuring "is-a" relationship, we promise that we need to use the superclass's  
constructor at first in the subclass's constructor. For example

```java
public class VengefulSLList<Item> extends SLList<Item>{
	SLList<Item> deletedItems;
	...
	public VengefulSLList(){
		super();
		deletedItems = new SLList<Item>();
	}
	...
}
```

This "super()" is explicitly using the constructor of  superclass, if we don't use it, the Java will automatically call the superclass's  
**non-argument constructor** for us. If we need to pass some arguments to the superclass's constructor, it must be use explicitly.

Note: you can implement multiple interface like that `public class ArrayDeque<T> implements Deque<T>,Iteratable<T>`​

### the object class

Every class in Java is descendant of Object class, or extends the Object class.

They may don't explicitly express the relationship with Object but they actually do it.

So, we want to know what we actually inherit from the Object class.

**Object method**

```java
String toString();
boolean equals(Object obj);
```

**toString()**

The default method of toString() is to print the location of the object in memory. For most of class we define by ourselves, we should  write our own toString() function to make the print result readable.

We note that the method `System.out.print()`​ will call the toString() method for all of argument and print the string.

```java
// the true situation
String s=myobject.toString();
System.out.print(s);
```

I will implement the toString() method in my ArreySet.java class. You can find this file in the JavaTest file.

**equals()**

We already have the "A==B" in Java and it will compare each bit in A and B. For the primitive variable it will compare their value and for the reference variable it will compare their address.

```java
public class test{
	private int a;
	private int b;
	public test(int a,int b){...}
	...
}
...
//main 
test A=test(1,2);
test B=test(1,2);
//However, A==B is false
```

For reference type variable, "==" means they point at the same object rather the objects have the same content.  
So we need to have equals() method.

‍

### static type and dynamic type

****This part note is not organized well, I will rewrite it after.

When we call the method, we will turn to dynamic type. It is what we have discussed before. However, static type will also make  
the difference.

In fact, a object, whose static type is superclass and dyanmic type is subclass which make extends, it can't call the method  
its dynamic type added.

And when we don't call the method, the object always **express as its static type**.

All in all, static type is its type and decide what the object can do, meanwhile the dynamic type decide what the object actually do.

### higher order function

How to create a function variable.

We can create the interface to make it possible. (类似于cpp的仿函数）

```java
//定义
public interface IntUnaryFunction {
    int apply(int x);
}
//实现
public class TenX implements IntUnaryFunction {
    /* Returns ten times the argument. */
    public int apply(int x) {
        return 10 * x;
    }
}
//函数使用
public static int do_twice(IntUnaryFunction f, int x) {
    return f.apply(f.apply(x));
}
//实际调用
System.out.println(do_twice(new TenX(), 2));
```

### polymorphism

The main idea of polymorphism is "many methods". And polymorphism express that an object can be an instance of its own class, its superclass and its superclass's superclass...

### Java's alternative for operator overloading

**introduction**

In cpp, we can write **one function** to find the max element in an array easily, as long as the elements in the array have operator overload about the '>', '<' and '=='. Luckily, for most of the class that is comparable, coders will provide the operator overloading functions. This one function can cover most of situations.

However, in Java, we **can't make operator overloading**, so things will become a little complex.

At first, we want to write a compare function in each class to replace the ability of operator overloading, but it means all of class should coincidentally use the same compare function name. So the good way is to write an interface and let each class to implement it.

```java
public interface ourComparable{
	public int compareTo(Object o);
} 
```

All class that is comparable should implement this interface, and provide the compareTo function.

**Comparable**

In fact, Java has already provided a similar interface for us and it has been used in countless Java library

```java
public interface Comparable<T>{
	public int compareTo(T obj);
}
...
//an example to implement the generic interface
public class Dog implements Comparable<Dog>{
	...
	public int compareTo(Dog anoDog){
		...
	}
}
```

In this real, built-in interface, we find it uses generic and it can avoid some awful cast from Object to each class.

**Comparator**

Sometimes, an object may have many dimensions to compare, but it can only have one compareTo function if we only use the Comparable interface. So Java also offers us another interface.

```java
public interface Comparator<T>{
	public int compare(T o1,T o2);
}
```

The usual way to use the comparator is different from the comparable, we will create a static nested class in the class we want to make compare.

```java
public class Dog implements Comparable<Dog>{
	...
	private static class NameCompare implements Comparator<Dog>{
		public int compare(Dog dog1,Dog dog2){
				return dog1.name.compareTo(dog2.name);
		}
	}
	...
}
```

We note that the nested class is private here, which means it can't be instanced outside.

```java
// 在其他的类中
NameCompare a=new Dog().new NameCompare(); // warning it isn't allowed
```

So, in Java's generic programming we should avoid using > < = and we should use the compareTo() equals() method.
