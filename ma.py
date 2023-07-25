


mapper = {}

with open(r'D:\workship\anewlife\wscnnlstmchange\mappers\1mer.txt','r') as f:
    for x in f:
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        #通过指定分隔符对字符串进行分割并返回一个列表，默认分隔符为所有空字符，包括空格、换行(\n)、制表符(\t)等.
        line = x.strip().split()
        # lower() 方法转换字符串中所有大写字符为小写。
        #line[0]去掉指定的一个，第一个
        word = line[0].lower()
        vec = [float(item) for item in line[1:]]
        mapper[word] = vec
        #print(f.readlines())
        print(word)
        print(vec)
# def Load_mapper(mapperfile):
#     mapper = {}
#     with open(mapperfile,'r') as f:
#         print('13')
#         for x in f:
#             # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
#             #通过指定分隔符对字符串进行分割并返回一个列表，默认分隔符为所有空字符，包括空格、换行(\n)、制表符(\t)等.
#             line = x.strip().split()
#             # lower() 方法转换字符串中所有大写字符为小写。
#             #line[0]去掉指定的一个，第一个
#             word = line[0].lower()
#             vec = [float(item) for item in line[1:]]
#             mapper[word] = vec
#             print(vec)
#
# mapperfile.read()
# mapperfile.close()
#
# with open(r'D:\workship\anewlife\wscnnlstmchange\mappers\1mer.txt', 'r') as f:
#
#     print(f.readlines())
