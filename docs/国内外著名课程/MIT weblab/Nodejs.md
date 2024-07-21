# Node.js

## nodejs

服务器是一个机器，处理，接受数据，并且与多个客户端进行通信

当我们正式与服务端通信时需要以这样的格式：`protocol://domain:port`​ 协议+域名+端口

当我们平时访问网站的时候却往往不需要加上端口，是因为我们的机器已经预留好了一系列的端口来处理例如HTTP协议

我们之前运行js代码的时候都是依赖于浏览器，而node.js可以使我们本地的机器运行js代码，它使得我们的机器可以运行js代码，从而让我们的机器也可以成为一台服务器

npm yarn 都是node.js的包管理工具

* 我们会把前端react的代码，放在client/
* 把服务端的代码放在server/

### 异步IO

node.js 是单线程的，通过事件队列的形式实现并发，当网络或文件请求发生时，Node.js 会注册一个回调函数，然后就去处理下一个请求了。当之前的操作完成后，会触发之前注册的回调函数，进而响应之前的请求。基于这个特性，node的许多操作都是异步的。

### js模块化

node.js 将js代码进行了更严格的模块化处理，使得其构建大型项目成为了可能

**commonJs**

我们的一个文件一般都有引用和被引用部分，我们用`module.exports`​和`require`​分别来处理可被引用的接口和引用其他文件的工作

例如

```javascript
// a.js
var name = 'morrain'
var age = 18
module.exports.name = name
module.exports.getAge = function(){
return age
}
//b.js
var a = require('./a.js')
console.log(a.name) // 'morrain'
console.log(a.getAge())// 18
```

**ECMAScript**

现代版本中有新的处理方式，使用`import`​和`export`​来处理

```javascript
import transform from './transform.js' /* default import */
import { var1 } from './consts.js' /* import a specific item */
import('http://example.com/example-module.js').then(() => {console.log('loaded')})
export const MODE = 'production' /* exported const */
```

## typescript

js是灵活的动态语言，这也导致了其检查非常不严格，容易出错，因此我们有TypeScript，让js的每一个变量和函数都和C等语言一样有类型定义。

TypeScript的作用是在编译期进行类型检查，发现错误，但最后使用的时候依旧会转换成js使用，在开发较为大型的项目的时候，使用typescript替代原生js往往是更好的选择

* todo
