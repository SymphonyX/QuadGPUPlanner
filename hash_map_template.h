#ifndef HASH_MAP_H
#define HASH_MAP_H

#include <stdint.h>
#include "common_template.h"

namespace CUDASTL{

// MemPool
// a simple memory pool, allows malloc, no free
// programmer should ensure that the memory used would never exceed the total amount
	class MemPool{
	public:
		__device__ void * malloc(uint32_t size);
	public:
		char * base_ptr;
		uint32_t total_size;
		uint32_t bytes_used;
	};

	__device__ void * MemPool::malloc(uint32_t size){
		uint32_t offset=ADD32(&bytes_used,size);
		return base_ptr+offset;
	}
	__host__ MemPool * CreateMemPool(uint32_t size){
		MemPool h_pool;
		cudaMalloc( (void**)(&(h_pool.base_ptr)), size);
		h_pool.total_size=size;
		h_pool.bytes_used=0;
		MemPool * d_ptr=CreateDeviceVar(h_pool);
		return d_ptr;
	}
	__host__ void DestroyMemPool(MemPool * p){
		MemPool h=DownloadDeviceVar(p);
		cudaFree(h.base_ptr);
		cudaFree(p);
	}

// UniqueList
// a list that only allows appending at the end, no delete, no insert at other places
// no duplicated lements allowed in the list
	template <class T>
	struct UniqueListNode{
		T data;
		UniqueListNode<T> * next;	
	};

	template<class T>
	class UniqueList{
	public:
		__device__ void init(MemPool * p);
		__device__ T & insert(const T & d);
		
	public:
		UniqueListNode<T> * head;
		MemPool * mem_pool;
	};

	template<class T>
	__device__ void UniqueList<T>::init(MemPool * p){
		head=NULL;
		mem_pool=p;
	}
	template<class T>
	__device__ T & UniqueList<T>::insert(const T & d){
		UniqueListNode<T> * node=(UniqueListNode<T> *) mem_pool->malloc(sizeof(UniqueListNode<T>));
		node->data=d;
		node->next=NULL;
		UniqueListNode<T> * curr=head;
		// if empty list
		if(curr==NULL){
			if(CASPTR(&head,NULL,node))
				return node->data;
		}
		curr=head;
		// list not empty
		while(1){
			if(curr->data==d)
				return curr->data;
			if(curr->next!=NULL){
				curr=curr->next;
				continue;
			}
			if(CASPTR(&(curr->next),NULL,node)){
				return node->data;
			}
		}
	}

// HashFunc
// hash function, maps argument type to uint32_t
	template <class T>
	class HashFunc{
	public:
		__device__ uint32_t operator()(const T & arg) const{
			return *(uint32_t *)(&arg);
		}
	};

	typedef const char * cstr_t;
	template<>
	class HashFunc<cstr_t>{
	public:
		__device__ uint32_t operator () (const cstr_t  &arg){
			cstr_t p=arg;
			uint32_t hashval;
			for (hashval = 0; *p != '\0'; p++)
				hashval = *p + 31 * hashval;
			return hashval;
		}
	};
	template<>
	class HashFunc<uint32_t>{
	public:
		__device__ uint32_t operator() (const uint32_t & arg){
			uint32_t key=arg;
			key += (key << 12);
			key ^= (key >> 22);
			key += (key << 4);
			key ^= (key >> 9);
			key += (key << 10);
			key ^= (key >> 2);
			key += (key << 7);
			key ^= (key >> 12);
			return key;
		}
	};
	
// HashMap
// a hash map that allows concurrent insert and read, but no delete
// user should specify a memory pool, from which the nodes will be allocated
// the memory pool should be large enough to hold all the nodes
	template<class key_t, class value_t>
	struct HashMapNode{
		key_t key;
		value_t value;
		__device__ __host__ bool operator == (const HashMapNode & rhs){return key==rhs.key;};
	};

	template <class key_t, class value_t, class hash_t=HashFunc<key_t> >
	class HashMap{
		typedef HashMapNode<key_t,value_t> MapNode;
	public:
		class Iterator{
		public:
			__device__ Iterator(){ptr=NULL;map=NULL;curr_bucket=0;}
			__device__ Iterator(const Iterator & rhs){ptr=rhs.ptr;map=rhs.map;curr_bucket=rhs.curr_bucket;}
			__device__ Iterator(UniqueListNode<MapNode> * node, HashMap<key_t,value_t,hash_t> * m, uint32_t bucket=0){ptr=node;map=m;curr_bucket=bucket;}

			__device__ Iterator & operator = (const Iterator & rhs){ptr=rhs.ptr;map=rhs.map;curr_bucket=rhs.curr_bucket;return *this;}
			__device__ bool operator == (const Iterator & rhs){return ptr==rhs.ptr;}
			__device__ bool operator != (const Iterator & rhs){return ptr!=rhs.ptr;}
			__device__ Iterator & operator ++ (){
				if(ptr==NULL)
					return *this;
				ptr=ptr->next;
				while(1){
					// at the middle of a bucket
					if(ptr!=NULL){
						return *this;
					}
					else{
						// at the end of the whole hash table
						if(curr_bucket==map->num_buckets-1){
							ptr=NULL;
							return *this;
						}
						// at the end of bucket, but middle of the hash table
						else{
							curr_bucket++;
							ptr=map->buckets[curr_bucket].head;
						}
					}
				}
			}
			__device__ Iterator operator ++ (int){Iterator old=*this;++ptr;return old;}
			__device__ MapNode & operator * (){return ptr->data;}
			__device__ MapNode * operator -> (){return &(ptr->data);}
		public:
			hash_t hash_func;
			UniqueListNode<MapNode> * ptr;
			HashMap * map;
			uint32_t curr_bucket;
		};
	public:
		__device__ value_t & operator[](const key_t & key);
		__device__ value_t * valueForKey(const key_t & key);

		__device__ Iterator begin();
		__device__ Iterator end();
		__device__ uint32_t hash(const key_t & key);
	public:
		hash_t hash_func;	
		MemPool * mem_pool;
		uint32_t num_buckets;
		UniqueList<MapNode> * buckets;
	};

	template<class key_t, class value_t, class hash_t>
	__device__ value_t & HashMap<key_t,value_t,hash_t>::operator[](const key_t & key){
		uint32_t index=hash(key);
		MapNode node;
		node.key=key;
		node.value=value_t();
		return (buckets[index].insert(node)).value;	// UniqueList.insert will return the MapNode if it already exists
	}

	template<class key_t, class value_t, class hash_t>
	__device__ value_t * HashMap<key_t, value_t, hash_t>::valueForKey(const key_t & key) {
		uint32_t index=hash(key);
		UniqueListNode<HashMapNode<key_t, value_t>> * node = buckets[index].head;
		while (1) {
			if (node->data.key == key) {
				return &node->data.value;
			}

			if (node->next == NULL) { break; }
			node = node->next;
		}
		return NULL;
	}

	template<class key_t, class value_t, class hash_t>
	__device__ HashMap<key_t, value_t, hash_t>::Iterator HashMap<key_t,value_t,hash_t>::begin(){
		return Iterator(buckets[0].head,this);
	}
	template<class key_t, class value_t, class hash_t>
	__device__ HashMap<key_t, value_t, hash_t>::Iterator HashMap<key_t,value_t,hash_t>::end(){
		return Iterator();
	}
	template<class key_t, class value_t, class hash_t>
	__device__ uint32_t HashMap<key_t,value_t,hash_t>::hash(const key_t & _key){
		return hash_func(_key)%num_buckets;
/*		uint32_t key=_key;
			key += (key << 12);
			key ^= (key >> 22);
			key += (key << 4);
			key ^= (key >> 9);
			key += (key << 10);
			key ^= (key >> 2);
			key += (key << 7);
			key ^= (key >> 12);
		return key%num_buckets;
*/	}

	//initialzers for hash map
	template<class key_t, class value_t, class hash_t>
	__global__ void InitBuckets (HashMap<key_t,value_t,hash_t> * hash_map) {
		if(get_thread_id()!=0)	// only one thread allowed to do this
			return;
		for(int i=0;i<hash_map->num_buckets;i++)
			hash_map->buckets[i].init(hash_map->mem_pool);
	}

	template<class key_t, class value_t, class hash_t>
	__host__ HashMap<key_t,value_t,hash_t> * CreateHashMap(uint32_t num_buckets, uint32_t num_elements){
		HashMap<key_t,value_t,hash_t> h;
		// calculate required memory size, and setup memory pool
		uint32_t mem_size= sizeof(UniqueListNode<HashMapNode<key_t,value_t> >)*num_elements;
		h.mem_pool=CreateMemPool(mem_size);
		// set number of buckets
		h.num_buckets=num_buckets;
		// allocate the buckets
		cudaMalloc((void **)(&(h.buckets)),sizeof(UniqueList<HashMapNode<key_t,value_t> >)*num_buckets);
		// create the device instance
		HashMap<key_t,value_t,hash_t> * d_ptr=CreateDeviceVar(h);
		// setup the buckets
		InitBuckets<key_t,value_t,hash_t> <<<1,1>>> (d_ptr);
		return d_ptr;
	}
	
	template<class key_t, class value_t, class hash_t>
	__host__ void DestroyHashMap(HashMap<key_t,value_t,hash_t> * d_ptr){
		HashMap<key_t,value_t,hash_t> h=DownloadDeviceVar(d_ptr);
		DestroyMemPool(h.mem_pool);
		cudaFree(h.buckets);
		cudaFree(d_ptr);
	}
};

#endif
