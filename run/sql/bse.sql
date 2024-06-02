CREATE TABLE bse
(
    id           bigint      NOT NULL AUTO_INCREMENT COMMENT '主键',
    sample_size  int         NOT NULL DEFAULT 0 COMMENT '样本记录数',
    sample_ratio double      NOT NULL DEFAULT 0.0 COMMENT '比例',
    k            double      NOT NULL DEFAULT 0 COMMENT 'k',
    p0_qe        double      NOT NULL DEFAULT 0.0 COMMENT '',
    p1_qe        double      NOT NULL DEFAULT 0.0 COMMENT '',
    run_id       varchar(64) NOT NULL DEFAULT '' COMMENT '运行id',
    tag          varchar(10) NOT NULL DEFAULT '' COMMENT '标记',
    create_time  timestamp     NOT NULL DEFAULT current_timestamp COMMENT '创建时间',
    PRIMARY KEY (id)
) ENGINE=innoDB DEFAULT CHARSET=utf8 comment 'bse';
