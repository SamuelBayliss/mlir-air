# Proposal - Remove partitions, and use tokens to encode concurrency. 

We currently use tokens to encode "happens after" relations in our code. 

Placing herds in different partitions places them in different "resource sets". This is intended to force concurrency. 

But we actually have several use-cases

| | Shares Resources | Different Resources |
| Shares Timeslot| X | Concurrency |
| Different Timeslot | Reclaim | Dependency |

So herds with dependent tokens imply different timeslots and herds that have a producer-consumer relationship using a channel may 
require that they are placed in different partitions ( resource sets). 

So we currently express the concept of conflicting timeslots ( including a relation on those timeslots ) using tokens, and the concept of
conflicting resource sets ( with no implied 'relation' on those resources-slots) differently. 

What if we unified this with relations on both time and space ( and got rid of partitions) 

%x = air.herd #air.schedule< after [%p0, %p1]> #air.resource< conflict [ %p0 ] > :  () {

} 

%x = air.herd #air.schedule< concurrent [%p1]> #air.resource< conflict [ %p0 ] > :  () {

} 

%x = air.herd #air.schedule< concurrent [%p1]> #air.resource< pool [ %p0 ] > :  () {

} 

Here an attribute on the herd describes the set ( schedule or resource) , a relation ( after, concurrent, conflict, or pool) and a set of 
other 'tokens'. You can view this as "mytoken" happens "after" {set-of-other-tokens}


