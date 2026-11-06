**字母异位词分组 strs = ["eat", "tea", "tan", "ate", "nat", "bat"]---->[["bat"],["nat","tan"],["ate","eat","tea"]]**  
根据指纹--[0]*26 设定group={}去做 最后返回list(group.values())

**最长连续序列 nums = [100,4,200,1,3,2] 输出：4**
用集合排序，然后取出每次的最长序列数 前-后=1 就加 否则就重新开始 cur=1

**移动0 nums = `[0,1,0,3,12]` 输出[1,3,12,0,0]**
从前往后赋值，把非零的从前往后赋值 j=0 然后在一个i里面随着i变化  然后再把j后面的赋值为0

**无重复字符的最长子串 s = "abcabcbb" 最长子串是 `"abc"` 输出3**
采用集合和左右指针去做  用while,right和left都是代表索引

**找到字符串中所有字母异位词 s = "cbaebabacd", p = "abc" 输出[0,6]**
使用2个指纹，s的指纹加一个字符就减去最前面的字符去不断更新指纹  然后来依次比较

**和为 K 的子数组 nums = [1,1,1], k = 2 输出2**
无法使用子集，子集可以不连续，子数组必须连续，使用pre_count={0:1}前n项和去做,用pre-k in pre_count这个pre-k的意思是有多少个pre-pre_count的元素=k 可以理解成取的是pre-k之后的连续数据一直到pre的  这之间的就是k count+=pre_count[pre]

**最大子数组和 nums = [-2,1,-3,4,-1,2,1,-5,4] 连续子数组 [4,-1,2,1] 的和最大，为 6**
使用cur = max(i,cur+i)去更新最新的子数组最大---要么取当前值，要么取前面的序列  然后max(cur,result)即可

**合并区间 intervals = [ [1,3],[2,6],[8,10],[15,18]] 输出：[ [1,6],[8,10],[15,18]]**
先按第一个元素排序.sort  然后比较新加和之前的是否有重叠

**轮转数组 nums = [1,2,3,4,5,6,7], k = 3 输出[5,6,7,1,2,3,4]**
先保存前面的(n-k)个  然后在让k个往前赋值nums[i] = nums[n-k+i] 在把后k个赋值nums[i+k]=tmp[i]

**除了自身以外数组的乘积 nums = `[1,2,3,4]` 输出: [24,12,8,6]**
用left和right分别得到i个元素左右两边的乘积  left--前一个元素*前一个left，right同理

**矩阵置零 输入：matrix = [ [1,1,1],[1,0,1],[1,1,1]] 输出  [ [1,0,1],[0,0,0],[1,0,1]]**
取出为零的索引放到tmp中  然后该行列都赋值为0  行为0就是固定列 列同理

**旋转矩阵 matrix = [ [1,2,3],[4,5,6],[7,8,9]] 输出[ [7,4,1],[8,5,2],[9,6,3]]**
先转置再翻转行的顺序.reverse

**搜索二维矩阵  O(log(m* n))的复杂度**
采用二分法的思想,left=0 right=m* n-1循环是while left<=right: mid=(left+right)//2,然后要找到这个数num=martix 【mid//n][ mid%n]如果这个数大于target,就取左边right=mid-1 如果小于就去右侧left=mid+1

**搜索二维矩阵II**
采用移动的思想，找到任意一个角点(左上角【0][n-1]),然后如果数大了就往左边移动,如果小了就往下移动去找

**矩阵乘法**
取出两个矩阵的行列，看矩阵是否可以相乘，构造出要求的矩阵的行列result=[【0】* n2 for _ in range(m1)] 然后用三层循环(i,j,k)最后一层循环是for k in range(n1):result[i]【j】+=mat1[i]【k】* mat2【k][j]

**前k个高频元素**
先排序，在用字典来存储每个元素出现的次数，group.get按照每个键的数量排序，reverse=True代表降序排列

**找重复数 nums = [1,3,4,2,2]  输出2**
采用group字典去存储每个数字以及对应的次数，for i,j in group.items()其中i是数值,j是次数，题目假设只存在一个，那直接返回i即可

**岛屿问题 找到1的个数，成岛屿的1算1个整体**
用dfs去做，dfs(p,q)中是不符合的时候  if p>=m or p<0 or q>=n or q<0 or grid 【p] [q]== "0" 然后直接return

**腐烂的橘子  计算把新鲜橘子全部腐烂的最短时间** 
只能用BFS的思想一层一层去找，采用队列deque,然后for _ in range(len(q))---当前的一层，然后把这一层的橘子都腐烂，if都是满足返回的grid[nr][nl] == 1把这个赋值为2，同时count-=1,q.append((nr,nl))

**课程表  是要根据给到的数组规则去判断数字能否学完 而不是去判断数组里面的是否合法**
numCourses = 4  
prerequisites = [ [1,0], [2,0], [3,1], [3,2]]
graph = [ [1,2], [3], [3], [] ]  
indegree = [0,1,1,2] 
prerequisitesd的子数组只会有两个,根据prerequisitesd得到邻表(学完某门课程后，哪些课程依赖它)和入度,入度为0代表可以直接学习,用deque存储,用while去做学习,同时去不断更新学完的课程(更新入度),用count计数看count是否等于numCourses

**全排列**
用remain和origin的思想去做(保留和剔除)，for i in range(len(origin))这个循环,同时里面也是有一层dfs的,列表相加是[],剔除的时候是[:i] + [i+1:]

**子集**
每次都要从result和nums去抽取元素,所以会有两重循环,new_nums.append([i]+j),然后result+=new_nums

**电话号码**
用group={'2':'abc',}这个字典把对应关系都存储下来，然后也是从result去取出元素，同时tmp去取出group[i]的元素，这里是不保留旧组合，因此使用result=tmp

**组合总和  candidates = [2,3,6,7], target = 7 输出 [ [2,2,3],[7]]**
要用dfs(start,path,remain)去做,start代表从哪个下标开始选择数字，path代表当前已经选择的数字，remain代表距离目标值还差多少，for i in range(start,len(candidate))在这个循环更新new_path，dfs(i,new_path,remain-candidate[i])

**括号生成**
用dfs(path,left,right)去做,path代表当前已经生成的括号字符串,left代表已经使用的左括号数量,right代表已经使用的右括号的数量,if left< n,dfs(path+'(',left+1,right)和if right< left dfs(path+')',left,right+1)去做回溯

**单词搜索**
用dfs的思想去做,dfs(i,j,k)中i代表当前行,j代表当前列,k代表要找word中的第几个字符.然后在dfs中要写i,j的范围约束,错误的话就返回False,然后要写borad【i][j]!=word[k]不对就要返回False.然后当k== len(word)-1的时候就返回True.要把当前用到的board值存下来,并且把当前的board标记为#(代表已经用了)再使用found去进行四个位置的搜索or来连接。然后取出board的值继续在返回整个函数的found

**分割回文串**
用path== path[::-1]去判断，采用dfs(remain,origin)去做,if not origin就result.append(remain)然后返回,for循环是对origin进行切片所以是(1,len(origin)+1)，然后path=origin(:i)然后在回文串的判断里面去进行内部dfs需要去更新remain和origin

**乘积最大子数组**
用cur_max和cur_min去记录每次的最大值和最小值，cur_max = max(cur_max * num,num)，cur_min同理去更新值，当遇到负数的时候最大值和最小值会交换(大变小，小变大)

**在排序数组中查找元素的第一个和最后一个位置**
复杂度是o(log(n))采用二分法，这里要返回两个位置，所以需要两次二分，第一次找到最左侧，第二次找到最右侧，由于要保证是最左侧所以num== target还不够,还要往左找,right=mid-1,最右侧同理去更新result[1]

**搜索旋转排序数组**
依旧是二分的思想，但是需要判断mid的左边还是右边有序,if nums[left]< nums[mid]然后看这个target是否在这个有序的范围之内,if nums[left] <= target< nums[mid]那就取左侧right=mid-1反之取右侧，如果是右边有序依旧先看是否在这个范围之内if nums[mid]< target <=nums[right]如果在就取右侧left=mid+1反之取左侧  有目标值的二分的while都是<=不是<

**搜索旋转排序数组中的最小值**
也是二分的思想，如果nums[mid]>nums[right]就代表右侧会有最小值，这个时候left=mid+1,如果不是的话最小值就在左侧,right=mid他会一个一个去搜索的不用mid-1的原因是可能mid会取到最小值。这里是找值所以while用<没有等号

**字符串解码**  ---有点理解不了
当遇到数字isdigit的时候cur_num=cur_num* 10+ch 当遇到‘【'的时候需要用stack存储之前的(cur_str,cur_num)---数字是这次的，str是上次的，然后这两个要重置，当取到'】'的时候取出之前的stack.pop()然后cur_str=pre+cur_str* num--顺序不能变，在就是如果是字符就更新cur_str+=ch

**每日温度**  --有点理解不了
stack去储存索引，栈中温度单调递减，然后每次都加上i,while stack and temp>栈的最左侧温度:pre=stack.pop(),result[pre] = i-pre

**爬楼梯**
记住dp[i]=dp[i-1]+dp[i-2]如果可以走三步就还要+dp[i-3]，对应公式就是a,b=b,a+b如果是3就是a,b,c=b,c,a+b+c

**杨辉三角**
使用tmp去构建[1] * (i+1)然后for i in range(1,i) tmp[j]=result[i-1]【j-1】+result 【i-1][即可

**打家劫舍**
爬楼类似的思路，但是a代表到前前一间房子为止，最多能偷多少钱。 b是到前一间房子为止，最多能偷多少钱。那么a更新就是b,b更新就是max(b,a+i),i代表当前房子的钱

**完全平方数**
采用dp=[float('inf')]* (n+1)去记录每个i需要平方数的最小数，所以while就是j* j<=i 然后dp[i]=min(dp[i],dp[i-j* j]+1)

**零钱兑换**
思路跟完全平方数差不多，这里是需要从coins去拿，上一个是正整数。for coin in coins 如果coin>i就继续  小于的话才可能更新,dp[i]=min(dp[i],dp[i-coin]+1)

**单词拆分**
使用for j in range(i) 然后if dp[j] and s[j:i] in word_set保证前面的j个字符可以被拼成以及后面的j到i个字符也可以被拼出来

**最长递增子序列**
还是用dp然后for j in range(i)如果dp[i]大于所有的dp[j]那么dp[i]就是新的最大值,dp[i]=max(dp[i],dp[j]+1)右边的dp[i]代表之前的最大值，再和这个时候进行比较

**乘积最大子数组**
用cur_max和cur_min不断去更新当前的最大值和最小值，如果num是负数就要去交换最大值和最小值  然后cur_max=max(cur_max* num,num)是为了防止出现0 

**分割等和子集**
先要看是否是奇数，如果不是就无法分割直接pass了，状态方程是dp[j]=dp[j] or dp[j-num],然后在做的时候是倒序，并且j要取道num，因此循环就是for j in range(target,num-1,-1)

**卖股票**
用min_price和max_profit去不断更新最便宜的价格和最大的利润  所以一次循环就可以计算出来 if min_price<=price去更新价格,elif price-min_price>=max_profit去更新利润

**跳跃游戏**
用max_reach=max(max_reach,i+num[i])去更新当前i能到达的最远距离，如果i>max_reach就代表无法到达i

**跳跃游戏2**
`end`：按照当前已经计算的跳跃次数，最远能够到哪里，这里比1少循环一次len(nums)-1 然后后面加一次判断if i== end: end=max_reach就可以了

**划分字母区间**
用字典记录字母出现的最后位置，然后一次循环遍历，用start和end去记录这次切片的开始和结束位置，end=max(end,group[s[i]])去更新当前的最远位置，end== i代表走了这个位置 然后加入结果并且更新start=i+1

**不同路径**
from math import factorial 用这个导入阶乘 然后(factorial(m+n-2))/[(factorial(m-1))* (factorial(n-1))]就是这个公式

**路径最小和**
$\boxed{dp[i][j]=\min(dp[i-1][j],dp[i][j-1])+grid[i][j]}$
记住上面的公式去做，但是最外层的需要单独处理(公式要求i,j均大于等于1)，第一列和第一行的计算也简单等于前面一个dp+当前的grid

**最长回文串**
写一个expand去判断是否回文串，这个是从这个点往外去搜索的，tmp1=expand(i,i)代表奇数，tmp2=(i-1,i)是偶数的，然后每次计算完tmp1和tmp2去更新result 根据len(result),len(tmp1),len(tmp2)的大小去更新

**最长公共序列**
用dp去存储(m+1)* (n* 1)的每个字母比过之后当前的最长公共序列的长度，+1是为了应对空字符。然后用两层循环，如果text1[i-1] == text2[j-1]--必须要-1，那么dp[i]【j】=dp[i-1]【j-1】+1如果不相等就是从从附近的两个取最大值 [i-1]【j】，[i]【j-1】

**编辑距离**
跟上一个题目差不多，不过这里0层的不是0是i,要重新赋值，然后如果不相等的话还要加一个dp[i-1]【j-1】这三个取最小值

**下一个排列**
找到比当前这个排列组成的整数稍微大一点的排列，先用if nums[i] < nums[i+1]找到要替换的位置i,然后再去找i之后的数字哪个要替换---找到后面比这个大的数，然后交换。在用nums[i+1:]=nums[i+1:]【::-1】就得到了结果直接在if中返回 如果找不到就是降序排列，直接返回nums.reverse()

## 字典的操作
for num,cnt in group.items() 取出字典的两个元素
																			list(group.key())取出所有的
list(group.values())直接取出字典的值
sorted(gourp,key=group.get,reverse=True)中group.get根据每个 key 获取对应的 value,reverse=True把value按照逆序排列  ｛2:3}代表2这个数字出现了3次 value=3

## 链表

相交链表---让两个指针分别走完两段链表，如果有交点就会相交 pa=pa.next  pa=headB

反转链表---用一个空节点，然后让链表的指针依次指他 cur.next = pre然后慢慢遍历

回文链表---用数组存储链表的值，然后倒序数组nums[::-1]

排序链表---用数组存储链表的值，然后.sort再去新建排序好的链表

环形链表I II---用集合存储，如果集合存在重复就返回True或者head

合并两个升序列表---创建新列表，依次指向两个链表最小的元素 最后再指向剩下的链表(不需要比大小了)

两个链表的和---创建新链表，需要carry作为进位，然后注意l1和l2是否存在即可

删除链表中的倒数第k个节点---展开成数组，然后去掉要删除的那个元素，再重新建立链表即可

两两交换链表的节点---新建一个链表，然后画图去做，两两存在才可以交换,cur.next and cur.next.next  通过赋值去构建出要求的链表

## 二叉树

前中后序遍历---编写一个dfs函数，根据不同遍历的顺序写dfs的顺序即可

二叉树的深度---要么是左边的要么是右边的，return 1+max(left,right)

翻转二叉树---直接交换顺序，if not root return None

对称二叉树---需要写一个辅助函数same(p,q),当not q and not p时返回的是true **数值(p.val和q.val)不等于**以及两个谁先没有都返回false 最后return same(p.left,q.right) and same(p.right,q.left)

二叉树的直径---直径等于左右深度，写一个全局变量self.diameter然后跟每次计算的深度和求最大

二叉树的层序遍历---写一个dfs(node,depath)带有深度的前序遍历函数，当result的长度和depath一样的时候就加[]然后加元素(在result[depath]加)

将升序数组转为二叉搜索树---左子树所有节点小于根节点，右子树所有节点大于根节点。其实也就是根据前序遍历去做build_tree(left,right)，加入的数值是root=TreeNode(nums[mid]) 然后root.left=built_tree(left,mid-1)右侧同理

验证二叉搜索树---用前序遍历的思想把数据都取出来，然后要注意去重，二叉搜索树是严格递增sorted(set(nums))

二叉树搜索树中第k小的元素---用[]存储前序遍历二叉搜索树的数(已经是升序的,或者采取如任意一种遍历，然后sort),返回数组第k小的元素

二叉树的右试图---用层序遍历的思想去遍历右边的节点数值,然后再去遍历左边的

二叉树按照前序遍历展开为链表---先用前序遍历，然后去修改树，新建一个指针head去指向树节点，在for循环中要往右边赋值，最后要把赋值完的节点左右子树指向None

二叉树的前序+中序构造二叉树---也是根据前序遍历的思想，从前序遍历可以第一个元素就是根节点，中序可以根据这个根节点划分左右子树(需要拿到这个索引index)，然后去填左右节点，前序遍历的左侧要去掉第一个元素(left--[1:index+1])这里的个数要和中序一样(in--index)，右侧二者一样

二叉树的中序+后序---后序的最后一个postorder(-1)是根节点，跟上述一样要从中序获取index，中序和后序的左侧是一样的，右侧后序要去掉最后一个元素(right--[index:-1])，也是记住带进去的数组元素个数中序和后序是一致的

二叉树的路径之和---写一个check去计算当前所有的路径之和，def(root,total)用total+=root.val去更新计算出来的数值,check(root.left,total)右侧同理，然后用dfs去遍历每个子节点，直径和最大路径是可以直接返回的，故不需要dfs去做了

二叉树的最大路径---跟直径的思路差不多，但是total = left+right+root.val并且return是root.val+max(left,right)直径不需要考虑数值而这里是需要的所以把1换成了root.val，直径是return 1+max(left,right) 还有一个点就是这个二叉树是可以断的所以left = max(get_total(root,left),0)右边同理，而直径都是正数不需要考虑这个

二叉树的最近公共祖先---用后序遍历的思想去找，return left if left else right返回找到的节点，他是一层一层往上找的  --- 不太能理解 不过能背下来了
