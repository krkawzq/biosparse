# Storage

Storage 是scl用于管理内存的模块，用于模拟对齐内存的分配和生命周期的管理。

Storage是一个轻量的内存管理器，使用Arc<>被其他的Span持有。

Storage的内存布局如下：

Storage
buffer: void* 实际指向的堆内存地址
start: void* 去掉堆内存不对齐的头部部分的对齐内存地址
size: 对齐大小的整数倍的总分配大小（tail是冗余的）

Storage的内存分配策略：
1. 外部传入一个Vec<usize>，Vec的每个元素表示这个span需要多大的内存。
2. 对每个元素向上取整到align size。
3. 计算sum，得到需要分配的大小
4. 申请堆分配器分配 sum + alignsize*2 - 2的内存（保证无论head tail多大都一定能够被我们自己规整化）（头尾都需要padding，以允许mask load， 防止segment fault）
5. 返回一个Vec，每个元素是span需要的指针

提供泛型重载（不知道rust能不能重载 哈，可以写成宏）
不接受Vec而是直接接受list


Span

Span是scl用于管理一段内存的模块
Span内存布局：
storage: Option<Arc<Storage>>
data: void* 实际指向的堆内存地址
size: 元素个数
flags: usize bitset

flags中包含：
is view 用于标记这个span是否是一个view（是否持有内存，也就是是否持有Storage）
is aligned 用于标记这个span的data是否已经aligned
mutable 用于标记这个span是否是mutable的（是否可写）




## sparse

pub struct CSR<V, I> {
    pub values: Vec<Span<V>>,    // 每行独立 Span
    pub indices: Vec<Span<I>>,   // 每行独立 Span
    pub rows: I,
    pub cols: I,
    pub nnz: Cell<Option<I>>
}