import collections
from typing import List
import heapq


class Twitter:

    def __init__(self):
        self.relation_graph = collections.defaultdict(set)
        self.post = collections.defaultdict(list)
        self.timer = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.timer -= 1
        self.post[userId].append([self.timer, tweetId])

    def getNewsFeed(self, userId: int) -> List[int]:
        feeds = []
        for followee in self.relation_graph[userId]:
            feeds += self.post[followee]
        feeds += self.post[userId]
        return [feed for _, feed in heapq.nsmallest(10, feeds)]

    def follow(self, followerId: int, followeeId: int) -> None:
        self.relation_graph[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.relation_graph[followerId]:
            self.relation_graph[followerId].remove(followeeId)

# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
