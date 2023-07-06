### 1. 概况

在系统设计的题目当中，当面试官简单描述完题目之后，我们需要先问一些问题来澄清题目的细节。

#### 第一步：澄清 requirements

requirements 分为两类：

- functional requirements：功能性需求，比如这个系统整体的功能能力。

举一个最简单的例子，Tiny URL:

Given a URL, our service should generate a shorter and unique alias of it. This is called a short link.

这些内容面试官只会简单描述几句话，并不会把功能展开来说（情况很少），那么我们就要从以下几个角度去思考关于功能扩展的问题（有先后重要性排序）：

0. 展开功能的细节，比如说有几种用户？每种用户的操作逻辑？这个功能是自动的还是手动的？
1. 某个功能有没有其他限制：时间限制？空间限制？性能限制？
2. 发生冲突时（并发请求）会怎么解决？
3. 无效输入时会怎么处理？

这类问题能够很好的展开这个功能的具体描述。

- non-functional requirements：非功能性需求，比如这个系统需要有哪些非功能性的特性， 在系统设计当中，有几个非常重要的非功能性需求需要我们去提问，如果不问会导致后面的技术选型出现很大的问题。

1. availability：可用性，这个系统需要有多少的可用性？99.99%？99.999%？
2. consistency：一致性，这个系统需要有多少的一致性？强一致性？弱一致性？
3. latency：延迟，这个系统的延迟要求是多少？毫秒级？微秒级？
4. elasticity: 弹性，这个系统的弹性要求是多少？需要能够快速的扩容和缩容吗？
5. scalability：可扩展性，这个系统的可扩展性要求是多少？每秒钟的请求量？每天的请求量？
6. reliability：可靠性，这个系统的可靠性要求是多少？每天的 downtime 是多少？

#### 第二步：数据计算

在澄清了 requirements 之后，我们需要计算数据的大小，这个数据的大小是非常重要的，有几个定理需要我们时刻记住：

1. 80-20 rule：80% 的数据来自于 20% 的用户，这个定理在系统设计当中非常重要，因为我们需要把这 20% 的用户放在第一位，这 20% 的用户的体验是最重要的。

我们需要计算的数据有以下几个：

1. traffic estimates：流量估计，这个系统每天的流量是多少？从而计算出每秒钟的流量是多少？
2. storage estimates：存储估计，计算出这个系统每天的数据量是多少？总共要存储多久的数据，这样来计算出总共需要多少的存储空间。
3. bandwidth estimates：带宽估计，计算出这个系统每秒钟的读写流量是多少？从而计算出需要多少的带宽。

所以根据以上三点，一定要问清楚来算出每秒的 QPS，每秒的带宽和总共需要的存储量。

最后总结出一个这样的表格就是最好的：
New URLs 200/s
URL redirections 20K/s
Incoming data 100KB/s
Outgoing data 10MB/s
Storage for 5 years 15TB
Memory for cache 170GB

同时也要询问几个核心问题，这些一定要问：

1. read-heavy or write-heavy：读多还是写多？读写比例是多少？
2. hotspots：热点数据是什么？比如说有没有热点 URL？有没有热点用户？是否符合 80-20 定律？（80% 的数据来自于 20% 的用户）
3. peak vs average：峰值和平均值的比例是多少？比如说平均值是 100 QPS，峰值是 1000 QPS，那么就是 10 倍的差距
4. traffic growth：流量增长率是多少？比如说每年增长 20%？每月增长 10%？

#### 第三步：系统总布局设计

在澄清了 requirements 和计算好数据之后，我们需要设计系统的总体布局。

对于一个最基本的系统，一定是由至少 3 个部分组成的：

1. client：客户端，负责发送请求
2. application server：服务器，负责处理总体业务逻辑
3. database：数据库，负责存储数据

这三个部分是最基本的，但是在实际的系统设计当中，我们需要考虑很多的其他的部分，比如说：

1. cache：缓存，负责缓存一些热点数据
2. load balancer：负载均衡器，负责把流量均匀的分发到不同的服务器上
3. message queue：消息队列，负责异步的处理一些任务
4. monitor：监控，负责监控整个系统的运行情况

！！！铁律！！！

1. 那么一般情况下，在 client 和 application server 之间，我们一般都需要加入一个 load balancer，这个 load balancer 的作用是什么呢？它的作用是把流量均匀的分发到不同的服务器上，这样来提高整个系统的吞吐量。

2. 在 application server 和 database 之间，我们一般都需要加入一个 cache，这个 cache 的作用是什么呢？它的作用是把一些热点数据缓存起来，这样来提高整个系统的读取速度。

3. 在 application server 和 database 之间，我们一般都需要加入一个 message queue，这个 message queue 的作用是什么呢？它的作用是把一些耗时的任务异步的处理掉，这样来提高整个系统的吞吐量。

#### 第四步：数据库设计

在澄清了 requirements 和计算好数据之后，我们需要设计数据库的 schema。

这里，我们就简单设计一个最基本的数据库 schema，url 这个题的 schema 由以下几个部分组成：

1. user：用户表，负责存储用户的信息，重点有 user ID：user name，create time, last login time
2. url：URL 表，负责存储 URL 的信息，重点有 hash：origin URL，create time, expire time

简单设计好 schema 之后，我们需要考虑一下几个问题：

#### SQL or NoSQL：使用 SQL 还是 NoSQL？

SQL 尽管有 ACID (Atomicity, Consistency, Isolation, Durability)的特性，但是在实际的系统设计当中，我们一般都不会使用 SQL，而是使用 NoSQL，为什么呢？因为 NoSQL 有以下几个优点：

1. 存放 unstructured data，如果要存储一个复杂的数据，比如说一个 多层嵌套的 JSON，那么使用 NoSQL 会更加的方便，而 SQL 需要把这个复杂的数据拆分成多个表，然后再进行存储。
2. Horizontal scalability，NoSQL 可以很方便的进行水平扩展，可以很好的解决读写性能的问题，而 SQL 的水平扩展是非常困难的，在 SQL 中，更多的是进行垂直扩展。
3. High availability，NoSQL 可以很方便的进行主从复制，从而保证高可用性，而 SQL 的主从复制是非常困难的，需要进行很多的配置。
4. Low cost.

注意 SQL 有一个缺点：

1. SQL 的缺点是 low consistency，数据可能不是最新的，因为主从复制需要一定的时间，所以可能会出现数据不一致的情况。

##### 选用哪种 NoSQL？

NoSQL 有很多种，比如说 key-value, document, column, graph。

1. key-value：key-value 是最简单的一种 NoSQL，它的特点是每个 key 对应一个 value，这个 value 可以是一个字符串或者其他东西。很经典的例子就是 DynamoDB，Redis。

2. document：document 是一种比较复杂的 NoSQL，它的特点是每个 document 对应一个 key，这个 document 可以是一个复杂的数据结构，比如说一个 JSON。很经典的例子就是 MongoDB。

3. column：column 是一种比较复杂的 NoSQL，它的特点是每个 column 对应一个 key，这个 column 可以是一个复杂的数据结构，比如说一个 JSON。很经典的例子就是 Cassandra。

4. graph：graph 是一种比较复杂的 NoSQL，它的特点是每个 node 对应一个 key，这个 node 可以是一个复杂的数据结构，比如说一个 JSON。很经典的例子就是 Neo4j。

##### 怎么设计 replication 和 partition？

replication 和 partition 是 NoSQL 的两个重要的特性，replication 是指主从复制，partition 是指分片。

1. replication：replication 的作用是什么呢？它的作用是提高系统的可用性，因为主从复制之后，即使主节点挂掉了，从节点也可以继续提供服务。replication 的缺点是什么呢？它的缺点是数据不一致，因为主从复制需要一定的时间，所以可能会出现数据不一致的情况。

2. partition：partition 的作用是什么呢？它的作用是提高系统的读写性能，因为分片之后，每个节点只需要处理部分的数据，所以可以提高系统的读写性能。partition 的缺点是什么呢？它的缺点是数据不一致，因为分片之后，每个节点只有部分的数据，所以可能会出现数据不一致的情况。

##### 怎么设计 cache？

cache 是 NoSQL 的一个重要的特性，它的作用是提高系统的读取速度，因为 cache 会把一些热点数据缓存起来，所以可以提高系统的读取速度。

#### 第五步：API 设计

...

#### 第六步：系统架构细节设计

...
